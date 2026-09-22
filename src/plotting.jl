# --- GPU-accelerated heatmap rendering ---
#
# Ported from epezent/implot#254 (backends branch). Instead of using ImPlot's
# CPU-side PlotHeatmap, we render colormapped data on the GPU via an FBO and
# display the result with ImPlot.PlotImage, preserving full axes/zoom/pan.
#
# Architecture:
#   HeatmapContext (module singleton) — shared GPU resources: shaders, colormap
#       texture, and a fullscreen quad. Lazily initialized on first matrix plot.
#   GPUHeatmap (per-plot) — data texture, colormapped output texture, and FBO.
#
# Pipeline per frame (when data changes):
#   1. Upload matrix data to a single-channel 2D texture (R32F / R32I / etc.)
#   2. Render a fullscreen quad into the FBO, sampling the data texture and a
#      1D colormap texture to produce an RGBA output texture
#   3. Display the output texture via ImPlot.PlotImage

# --- Shaders ---
#
# The vertex shader draws a fullscreen quad (two triangles). The fragment shader
# normalizes the heatmap value to [0,1] using min/max uniforms, then samples a
# 1D colormap texture. Two fragment variants exist: one for float data
# (sampler2D) and one for integer data (isampler2D).

const HEATMAP_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 Position;
layout (location = 1) in vec2 UV;
out vec2 Frag_UV;
void main() {
    Frag_UV = UV;
    gl_Position = vec4(Position, 0.0, 1.0);
}
"""

# Body shared between the float and integer fragment shaders. The caller
# provides the sampler type (sampler2D vs isampler2D) and whether to handle NaN
# (only float samples can be NaN); everything after that — log10 remap,
# colormap lookup with half-texel inset — is identical. In log mode
# min_val/max_val are already log10'd on the CPU; non-positive samples have no
# real log and become transparent.
function heatmap_fragment_source(sampler::String, handle_nan::Bool)
    nan_block = handle_nan ? """
    if (isnan(value)) {
        Out_Color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
""" : ""
    """
    #version 330 core
    precision mediump float;
    in vec2 Frag_UV;
    out vec4 Out_Color;
    uniform sampler1D colormap;
    uniform $(sampler) heatmap;
    uniform float min_val;
    uniform float max_val;
    uniform bool use_log;
    void main() {
        float value = float(texture(heatmap, Frag_UV).r);
    $(nan_block)
        if (use_log) {
            if (value <= 0.0) {
                Out_Color = vec4(0.0, 0.0, 0.0, 0.0);
                return;
            }
            value = log(value) / log(10.0);
        }
        // Half-texel inset avoids sampling beyond the colormap edges
        float min_tex_offs = 0.5 / float(textureSize(colormap, 0));
        float offset = (value - min_val) / (max_val - min_val);
        offset = mix(min_tex_offs, 1.0 - min_tex_offs, clamp(offset, 0.0, 1.0));
        Out_Color = texture(colormap, offset);
    }
    """
end

const HEATMAP_FRAGMENT_FLOAT = heatmap_fragment_source("sampler2D", true)
const HEATMAP_FRAGMENT_INT = heatmap_fragment_source("isampler2D", false)

# --- GL helpers ---

"""Compile a GLSL shader from source, raising on error."""
function compile_shader(source::String, type::GLenum)
    shader = glCreateShader(type)
    glShaderSource(shader, 1, Ref(pointer(source)), C_NULL)
    glCompileShader(shader)
    status = Ref{GLint}(0)
    glGetShaderiv(shader, GL_COMPILE_STATUS, status)
    if status[] != GL_TRUE
        log_len = Ref{GLint}(0)
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, log_len)
        log_buf = Vector{UInt8}(undef, log_len[])
        glGetShaderInfoLog(shader, log_len[], C_NULL, pointer(log_buf))
        error("Shader compilation failed: $(String(log_buf))")
    end
    return shader
end

"""Link a vertex + fragment shader into a program, raising on error."""
function link_program(vertex::GLuint, fragment::GLuint)
    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    status = Ref{GLint}(0)
    glGetProgramiv(program, GL_LINK_STATUS, status)
    if status[] != GL_TRUE
        log_len = Ref{GLint}(0)
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, log_len)
        log_buf = Vector{UInt8}(undef, log_len[])
        glGetProgramInfoLog(program, log_len[], C_NULL, pointer(log_buf))
        error("Program link failed: $(String(log_buf))")
    end
    return program
end

# --- Shared GPU state (module singleton) ---

"""
Shared GPU resources for heatmap rendering, created once and reused across all
plots. Contains compiled shader programs (float + integer variants), a 1D
colormap texture sampled from ImPlot, and a fullscreen quad VAO/VBO.
"""
mutable struct HeatmapContext
    # Shader programs — float variant uses sampler2D, int uses isampler2D
    shader_float::GLuint
    shader_int::GLuint

    # Uniform locations for each shader variant
    loc_min_float::GLint
    loc_max_float::GLint
    loc_heatmap_float::GLint
    loc_colormap_float::GLint
    loc_log_float::GLint

    loc_min_int::GLint
    loc_max_int::GLint
    loc_heatmap_int::GLint
    loc_colormap_int::GLint
    loc_log_int::GLint

    # 1D RGBA8 texture (256 entries) built from ImPlot's active colormap
    colormap_tex::GLuint
    # Which ImPlot colormap is uploaded (-2 = none; -1 is a valid ImPlot value)
    colormap_id::Int

    # Fullscreen quad geometry for FBO rendering
    vao::GLuint
    vbo::GLuint
end

function create_heatmap_context()
    # Compile vertex shader (shared between float and int variants)
    vert = compile_shader(HEATMAP_VERTEX_SHADER, GL_VERTEX_SHADER)

    frag_f = compile_shader(HEATMAP_FRAGMENT_FLOAT, GL_FRAGMENT_SHADER)
    shader_float = link_program(vert, frag_f)
    glDeleteShader(frag_f)

    frag_i = compile_shader(HEATMAP_FRAGMENT_INT, GL_FRAGMENT_SHADER)
    shader_int = link_program(vert, frag_i)
    glDeleteShader(frag_i)

    glDeleteShader(vert)

    # Cache uniform locations for both shader variants
    loc_min_float = glGetUniformLocation(shader_float, "min_val")
    loc_max_float = glGetUniformLocation(shader_float, "max_val")
    loc_heatmap_float = glGetUniformLocation(shader_float, "heatmap")
    loc_colormap_float = glGetUniformLocation(shader_float, "colormap")
    loc_log_float = glGetUniformLocation(shader_float, "use_log")

    loc_min_int = glGetUniformLocation(shader_int, "min_val")
    loc_max_int = glGetUniformLocation(shader_int, "max_val")
    loc_heatmap_int = glGetUniformLocation(shader_int, "heatmap")
    loc_colormap_int = glGetUniformLocation(shader_int, "colormap")
    loc_log_int = glGetUniformLocation(shader_int, "use_log")

    # Allocate colormap texture (filled lazily by update_colormap!)
    colormap_tex_ref = Ref{GLuint}(0)
    glGenTextures(1, colormap_tex_ref)
    colormap_tex = colormap_tex_ref[]

    # Build a fullscreen quad: two triangles covering [-1,1] in clip space.
    # UVs are transposed (u↔v swapped) so that the first matrix dim maps to
    # the vertical axis and the second to the horizontal, matching matplotlib
    # (data[1,1] at top-left, data[rows,cols] at bottom-right). The data texture
    # is uploaded as-is (Julia-column-major → texture scanline), so the shader
    # samples data_tex(v, u) to get data[i=u_in_pixels+1, j=v_in_pixels+1]
    # effectively transposed here via the UV swap.
    #   Each vertex: (x, y, u, v)
    quad_vertices = Float32[
        -1, -1, 0, 0,  # bottom-left
         1, -1, 0, 1,  # bottom-right
        -1,  1, 1, 0,  # top-left
         1, -1, 0, 1,  # bottom-right
         1,  1, 1, 1,  # top-right
        -1,  1, 1, 0,  # top-left
    ]

    vao_ref = Ref{GLuint}(0)
    vbo_ref = Ref{GLuint}(0)
    glGenVertexArrays(1, vao_ref)
    glGenBuffers(1, vbo_ref)
    vao = vao_ref[]
    vbo = vbo_ref[]

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW)
    stride = 4 * sizeof(Float32)
    # layout(location = 0) — position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, C_NULL)
    glEnableVertexAttribArray(0)
    # layout(location = 1) — UV
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(2 * sizeof(Float32)))
    glEnableVertexAttribArray(1)
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return HeatmapContext(
        shader_float, shader_int,
        loc_min_float, loc_max_float, loc_heatmap_float, loc_colormap_float, loc_log_float,
        loc_min_int, loc_max_int, loc_heatmap_int, loc_colormap_int, loc_log_int,
        colormap_tex, -2,
        vao, vbo,
    )
end

function destroy!(ctx::HeatmapContext)
    glDeleteProgram(ctx.shader_float)
    glDeleteProgram(ctx.shader_int)
    tex_ref = Ref(ctx.colormap_tex)
    glDeleteTextures(1, tex_ref)
    vao_ref = Ref(ctx.vao)
    vbo_ref = Ref(ctx.vbo)
    glDeleteVertexArrays(1, vao_ref)
    glDeleteBuffers(1, vbo_ref)
end

# Google's Turbo colormap as packed ABGR colors, from
# https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
const TURBO_COLORMAP_DATA = ig.ImU32[
    0xff3b1230, 0xff431532, 0xff4a1833, 0xff511b34, 0xff581e35, 0xff5f2136, 0xff662437, 0xff6d2738,
    0xff732a39, 0xff792d3a, 0xff802f3b, 0xff86323c, 0xff8b353d, 0xff91383e, 0xff973b3f, 0xff9c3e3f,
    0xffa24040, 0xffa74341, 0xffac4641, 0xffb14942, 0xffb54b42, 0xffba4e43, 0xffbf5144, 0xffc35444,
    0xffc75644, 0xffcb5945, 0xffcf5c45, 0xffd35e45, 0xffd66146, 0xffda6446, 0xffdd6646, 0xffe06946,
    0xffe36b46, 0xffe66e47, 0xffe97147, 0xffeb7347, 0xffee7647, 0xfff07847, 0xfff27b47, 0xfff47d46,
    0xfff68046, 0xfff88246, 0xfffa8546, 0xfffb8746, 0xfffc8a45, 0xfffd8c45, 0xfffe8f44, 0xfffe9143,
    0xffff9442, 0xffff9641, 0xffff9940, 0xfffe9b3e, 0xfffe9e3d, 0xfffda03b, 0xfffca33a, 0xfffba538,
    0xfffaa837, 0xfff8ab35, 0xfff7ad33, 0xfff5af31, 0xfff4b22f, 0xfff2b42e, 0xfff0b72c, 0xffeeb92a,
    0xffebbc28, 0xffe9be27, 0xffe7c025, 0xffe4c323, 0xffe2c522, 0xffdfc720, 0xffddc91f, 0xffdacb1e,
    0xffd8cd1c, 0xffd5d01b, 0xffd2d21a, 0xffd0d41a, 0xffcdd519, 0xffcad718, 0xffc8d918, 0xffc5db18,
    0xffc2dd18, 0xffc0de18, 0xffbde018, 0xffbbe219, 0xffb9e319, 0xffb6e41a, 0xffb4e61c, 0xffb2e71d,
    0xffafe91f, 0xffacea20, 0xffaaeb22, 0xffa7ec25, 0xffa4ee27, 0xffa1ef2a, 0xff9ef02c, 0xff9bf12f,
    0xff98f232, 0xff94f335, 0xff91f438, 0xff8ef53c, 0xff8af63f, 0xff87f743, 0xff84f846, 0xff80f84a,
    0xff7df94e, 0xff7afa52, 0xff76fa55, 0xff73fb59, 0xff6ffc5d, 0xff6cfc61, 0xff69fd65, 0xff66fd69,
    0xff62fe6d, 0xff5ffe71, 0xff5cfe75, 0xff59fe79, 0xff56ff7d, 0xff53ff80, 0xff51ff84, 0xff4eff88,
    0xff4bff8b, 0xff49ff8f, 0xff47ff92, 0xff44fe96, 0xff42fe99, 0xff40fe9c, 0xff3ffd9f, 0xff3dfda1,
    0xff3cfca4, 0xff3afca7, 0xff39fba9, 0xff38fbac, 0xff37faaf, 0xff36f9b1, 0xff36f8b4, 0xff35f7b7,
    0xff35f6b9, 0xff34f5bc, 0xff34f4be, 0xff34f3c1, 0xff34f1c3, 0xff34f0c6, 0xff34efc8, 0xff34edcb,
    0xff34eccd, 0xff34ead0, 0xff35e9d2, 0xff35e7d4, 0xff35e5d7, 0xff36e4d9, 0xff36e2db, 0xff37e0dd,
    0xff37dfdf, 0xff37dde1, 0xff38dbe3, 0xff38d9e5, 0xff39d7e7, 0xff39d5e9, 0xff39d3eb, 0xff3ad1ec,
    0xff3acfee, 0xff3acdef, 0xff3acbf1, 0xff3ac9f2, 0xff3ac7f4, 0xff3ac5f5, 0xff3ac3f6, 0xff3ac1f7,
    0xff39bef8, 0xff39bcf9, 0xff39bafa, 0xff38b8fb, 0xff37b6fb, 0xff36b3fc, 0xff36b1fc, 0xff35aefd,
    0xff34acfd, 0xff33a9fe, 0xff32a7fe, 0xff31a4fe, 0xff30a1fe, 0xff2f9efe, 0xff2d9bfe, 0xff2c99fe,
    0xff2b96fe, 0xff2a93fe, 0xff2990fe, 0xff278dfd, 0xff268afd, 0xff2587fc, 0xff2384fc, 0xff2281fb,
    0xff217efb, 0xff1f7bfa, 0xff1e78f9, 0xff1d75f9, 0xff1c72f8, 0xff1a6ff7, 0xff196cf6, 0xff1869f5,
    0xff1766f4, 0xff1563f3, 0xff1460f2, 0xff135df1, 0xff125bf0, 0xff1158ef, 0xff1055ed, 0xff0f53ec,
    0xff0e50eb, 0xff0d4eea, 0xff0c4be8, 0xff0c49e7, 0xff0b47e5, 0xff0a45e4, 0xff0a43e2, 0xff0941e1,
    0xff083fdf, 0xff083ddd, 0xff073bdc, 0xff0739da, 0xff0637d8, 0xff0635d6, 0xff0533d4, 0xff0531d2,
    0xff052fd0, 0xff042dce, 0xff042bcc, 0xff042aca, 0xff0328c8, 0xff0326c5, 0xff0325c3, 0xff0223c1,
    0xff0221be, 0xff0220bc, 0xff021eb9, 0xff021db7, 0xff011bb4, 0xff011ab2, 0xff0118af, 0xff0117ac,
    0xff0116a9, 0xff0114a7, 0xff0113a4, 0xff0112a1, 0xff01109e, 0xff010f9b, 0xff010e98, 0xff010d95,
    0xff010b92, 0xff010a8e, 0xff02098b, 0xff020888, 0xff020785, 0xff020681, 0xff02057e, 0xff03047a,
]

# The ImPlot index of the Turbo colormap, registering it on first use. The index
# lives in the ImPlot context, so it can't be cached in a global: AddColormap
# returns -1 (ImPlot's "current colormap") if the name is already taken.
function turbo_colormap()
    cmap = ImPlot.GetColormapIndex("Turbo")
    if cmap == -1
        cmap = ImPlot.AddColormap("Turbo", TURBO_COLORMAP_DATA,
                                  length(TURBO_COLORMAP_DATA), false)
    end
    return Cint(cmap)
end

"""
Re-upload the 1D colormap texture if the active ImPlot colormap has changed.
Samples 256 points from the colormap and uploads as GL_RGBA8 with linear
filtering (smooth gradient between color stops).
"""
function update_colormap!(ctx::HeatmapContext, cmap::Integer)
    ctx.colormap_id == cmap && return

    n = 256
    pixels = Vector{UInt8}(undef, n * 4)
    for i in 0:n-1
        t = i / (n - 1)
        col = ImPlot.SampleColormap(t, cmap)
        idx = i * 4
        pixels[idx + 1] = round(UInt8, clamp(col.x, 0, 1) * 255)
        pixels[idx + 2] = round(UInt8, clamp(col.y, 0, 1) * 255)
        pixels[idx + 3] = round(UInt8, clamp(col.z, 0, 1) * 255)
        pixels[idx + 4] = round(UInt8, clamp(col.w, 0, 1) * 255)
    end

    glBindTexture(GL_TEXTURE_1D, ctx.colormap_tex)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_1D, 0)

    ctx.colormap_id = cmap
end

# Lazily initialized module-level singleton
const _heatmap_ctx = Ref{Union{Nothing, HeatmapContext}}(nothing)

function get_heatmap_context()
    if isnothing(_heatmap_ctx[])
        _heatmap_ctx[] = create_heatmap_context()
    end
    return _heatmap_ctx[]
end

"""Destroy shared heatmap GPU resources. Call before tearing down the GL context."""
function destroy_heatmap_context!()
    if !isnothing(_heatmap_ctx[])
        destroy!(_heatmap_ctx[])
        _heatmap_ctx[] = nothing
    end
end

# --- Per-plot GPU state ---

"""
Per-plot GPU resources for heatmap rendering:
- `data_tex`:   single-channel 2D texture holding the raw matrix data
- `output_tex`: RGBA8 2D texture holding the colormapped result (fed to PlotImage)
- `fbo`:        framebuffer targeting output_tex for off-screen rendering
"""
mutable struct GPUHeatmap
    data_tex::GLuint
    output_tex::GLuint
    fbo::GLuint
    width::Int
    height::Int
    is_integer::Bool
    # Reusable buffer for data that needs conversion (e.g. Float64 → Float32).
    # Avoids allocating a new array every frame.
    convert_buf::Vector{UInt8}
    # Reused histogram bin counts for approximate 1st/99th percentile
    # estimation, avoiding a full copy + sort of the input.
    hist_buf::Vector{Int32}
    # Whether the texture was last rendered in log mode — toggling this in the
    # UI forces a re-render with fresh percentiles.
    log_scale::Bool
    # The colormap it was last rendered with.
    colormap::Cint
end

function GPUHeatmap()
    tex_refs = Ref{GLuint}(0)

    glGenTextures(1, tex_refs)
    data_tex = tex_refs[]

    glGenTextures(1, tex_refs)
    output_tex = tex_refs[]

    fbo_ref = Ref{GLuint}(0)
    glGenFramebuffers(1, fbo_ref)
    fbo = fbo_ref[]

    return GPUHeatmap(data_tex, output_tex, fbo, 0, 0, false, UInt8[], Int32[], false, -1)
end

function destroy!(h::GPUHeatmap)
    for tex in (h.data_tex, h.output_tex)
        tex_ref = Ref(tex)
        glDeleteTextures(1, tex_ref)
    end
    fbo_ref = Ref(h.fbo)
    glDeleteFramebuffers(1, fbo_ref)
end

# --- Data type mapping ---
#
# Maps Julia eltypes to GL format tuples:
#   (internal_format, pixel_format, pixel_type, is_integer)
# Types without direct GL equivalents (Float64, Int64, UInt64) are converted
# to their 32-bit counterparts by prepare_data() before upload.

gl_format(::Type{Float32}) = (GL_R32F,  GL_RED,         GL_FLOAT,          false)
gl_format(::Type{Int32})   = (GL_R32I,  GL_RED_INTEGER, GL_INT,            true)
gl_format(::Type{UInt32})  = (GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT,   true)
gl_format(::Type{Int16})   = (GL_R16I,  GL_RED_INTEGER, GL_SHORT,          true)
gl_format(::Type{UInt16})  = (GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, true)
gl_format(::Type{Int8})    = (GL_R8I,   GL_RED_INTEGER, GL_BYTE,           true)
gl_format(::Type{UInt8})   = (GL_R8UI,  GL_RED_INTEGER, GL_UNSIGNED_BYTE,  true)

# Target GL type for eltypes that need conversion
gl_convert_type(::Type{Float64}) = Float32
gl_convert_type(::Type{Int64})   = Int32
gl_convert_type(::Type{UInt64})  = UInt32
gl_convert_type(::Type)          = Float32  # fallback

# Types that can be uploaded directly without conversion
const GLNativeTypes = Union{Float32, Int32, UInt32, Int16, UInt16, Int8, UInt8}

"""
Materialize matrix data as a dense, GL-uploadable matrix. Returns the original
data zero-copy when it's already a native-eltype dense array, otherwise copies
(and type-converts if needed) into the reusable `convert_buf`.
"""
function prepare_data!(h::GPUHeatmap, data::AbstractMatrix{T}) where T
    base = data isa DimArray ? parent(data) : data
    if T <: GLNativeTypes && base isa DenseArray
        return base
    else
        # Copy into the cached byte buffer, converting to the GL-native type
        # when necessary.
        G = T <: GLNativeTypes ? T : gl_convert_type(T)
        nbytes = length(data) * sizeof(G)
        resize!(h.convert_buf, nbytes)
        buf = unsafe_wrap(Matrix{G}, Ptr{G}(pointer(h.convert_buf)), size(data))
        copyto!(buf, data)
        return buf
    end
end

"""
Upload matrix data to the GPU data texture. Converts to a GL-compatible type if
needed (reusing an internal buffer), then uploads as a single-channel 2D
texture. Resizes the output texture and re-attaches the FBO if dimensions changed.
"""
function upload_data!(h::GPUHeatmap, data::AbstractMatrix)
    gpu_data = prepare_data!(h, data)
    T = eltype(gpu_data)
    internal_fmt, pixel_fmt, pixel_type, is_integer = gl_format(T)

    rows, cols = size(gpu_data)
    h.is_integer = is_integer

    # Upload raw data to the single-channel data texture
    glBindTexture(GL_TEXTURE_2D, h.data_tex)
    # Julia matrices are column-major: each column of `rows` elements is
    # contiguous in memory.  OpenGL reads row-major (width elements per
    # scanline), so we pass rows as width so that each texture row reads
    # exactly one Julia column.  The resulting texture is the transpose of
    # the matrix: texture pixel (x, y) = data[x+1, y+1].
    glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, rows, cols, 0, pixel_fmt, pixel_type, gpu_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)

    # Resize the RGBA output texture and re-attach to FBO when dimensions change.
    # Output is the visually-oriented image (width=cols, height=rows) — the quad
    # UVs transpose the input while rendering.
    if h.width != cols || h.height != rows
        h.width = cols
        h.height = rows

        glBindTexture(GL_TEXTURE_2D, h.output_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cols, rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, C_NULL)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, h.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, h.output_tex, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    end
end

# Approximate (p1, p99) of `data` with no sorting. Samples every 10th element
# (the whole array for small inputs) and estimates the percentiles with a
# two-pass histogram: pass 1 gets the value range, pass 2 bins the sample, then
# we walk the cumulative counts to the target rank with linear in-bin
# interpolation. Non-finite samples are dropped (and non-positive ones in log
# mode, which have no real log). Log mode bins log10 of the samples, since
# uniform linear bins lump the low decades of wide-range data into the first
# bin. `buf` is the reused bin-count scratch.
# Returns `(0.0, 1.0)` if no usable samples; always returns finite, log10-space
# values when `log` is set (matching the colormap's domain).
const PCTILE_NBINS = 2048

# Min/max over the strided sample, dropping non-finite values (and non-positive
# ones in log mode, which have no real log). Returns (Inf, -Inf) if nothing
# qualifies.
function finite_extrema(data::AbstractMatrix, stride::Int, log::Bool)
    lo = Inf
    hi = -Inf
    n = length(data)

    @inbounds for i in 1:stride:n
        x = Float64(data[i])
        if log ? (isfinite(x) && x > 0) : isfinite(x)
            lo = ifelse(x < lo, x, lo)
            hi = ifelse(x > hi, x, hi)
        end
    end

    return (lo, hi)
end

function sampled_pctile!(buf::Vector{Int32}, data::AbstractMatrix, log::Bool=false)
    n = length(data)
    if n == 0
        return (0.0, 1.0)
    end
    stride = n < 1000 ? 1 : 10

    # Pass 1: value range over the valid sample.
    lo, hi = finite_extrema(data, stride, log)
    if !isfinite(lo) || !isfinite(hi)
        return (0.0, 1.0)
    end
    if log
        lo, hi = log10(lo), log10(hi)
    end
    if !(hi > lo)
        return (lo, lo)
    end

    # Pass 2: histogram the sample into PCTILE_NBINS uniform bins.
    if length(buf) != PCTILE_NBINS
        resize!(buf, PCTILE_NBINS)
    end
    fill!(buf, 0)
    scale = PCTILE_NBINS / (hi - lo)
    total = 0
    @inbounds for i in 1:stride:n
        x = Float64(data[i])
        if log ? (isfinite(x) && x > 0) : isfinite(x)
            v = log ? log10(x) : x
            b = clamp(floor(Int, (v - lo) * scale) + 1, 1, PCTILE_NBINS)
            buf[b] += Int32(1)
            total += 1
        end
    end

    if total == 0
        return (0.0, 1.0)
    end

    binwidth = (hi - lo) / PCTILE_NBINS
    p1 = quantile_at(buf, 0.01 * total, lo, hi, binwidth)
    p99 = quantile_at(buf, 0.99 * total, lo, hi, binwidth)
    return (p1, p99)
end

# Walk the histogram's cumulative counts to the target rank, interpolating
# within the straddling bin for a smoother estimate. Falls back to hi if the
# target is past the last count.
function quantile_at(buf::Vector{Int32}, target, lo, hi, binwidth)
    cum = 0
    @inbounds for b in 1:PCTILE_NBINS
        c = buf[b]
        if cum + c >= target
            frac = c > 0 ? (target - cum) / c : 0.0
            return lo + (b - 1 + frac) * binwidth
        end
        cum += c
    end
    return hi
end

"""
Render the colormapped heatmap into the output texture via the FBO. Binds the
data texture (unit 0) and colormap texture (unit 1), draws a fullscreen quad
with the appropriate shader, then restores the previous GL state so we don't
interfere with Dear ImGui's rendering.
"""
function render_colormapped!(h::GPUHeatmap, ctx::HeatmapContext, min_val, max_val, use_log::Bool)
    h.width == 0 && return

    # Save GL state that we'll modify (Dear ImGui expects these unchanged)
    prev_program = Ref{GLint}(0)
    prev_fbo = Ref{GLint}(0)
    prev_viewport = Vector{GLint}(undef, 4)
    glGetIntegerv(GL_CURRENT_PROGRAM, prev_program)
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, prev_fbo)
    glGetIntegerv(GL_VIEWPORT, prev_viewport)

    # Set up off-screen render target
    glBindFramebuffer(GL_FRAMEBUFFER, h.fbo)
    glViewport(0, 0, h.width, h.height)

    # Activate the appropriate shader and set uniforms
    if h.is_integer
        glUseProgram(ctx.shader_int)
        glUniform1f(ctx.loc_min_int, Float32(min_val))
        glUniform1f(ctx.loc_max_int, Float32(max_val))
        glUniform1i(ctx.loc_heatmap_int, 0)
        glUniform1i(ctx.loc_colormap_int, 1)
        glUniform1i(ctx.loc_log_int, use_log)
    else
        glUseProgram(ctx.shader_float)
        glUniform1f(ctx.loc_min_float, Float32(min_val))
        glUniform1f(ctx.loc_max_float, Float32(max_val))
        glUniform1i(ctx.loc_heatmap_float, 0)
        glUniform1i(ctx.loc_colormap_float, 1)
        glUniform1i(ctx.loc_log_float, use_log)
    end

    # Bind data texture to unit 0, colormap to unit 1
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, h.data_tex)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_1D, ctx.colormap_tex)

    # Render fullscreen quad
    glBindVertexArray(ctx.vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)
    glBindVertexArray(0)

    # Restore previous GL state
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_1D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo[])
    glUseProgram(prev_program[])
    glViewport(prev_viewport[1], prev_viewport[2], prev_viewport[3], prev_viewport[4])
end

# --- Plot struct with optional GPU heatmap ---

const FIT_TYPES = ["None", "Line", "Gaussian", "erf", "sin"]

# Per-parameter UI state: `fixed` selects whether the slot is held in the next
# fit; `value` is the committed value used by the fit; `edit_buf` is the
# InputDouble binding, copied into `value` only on Enter so live keystrokes
# don't drive the fit.
@kwdef mutable struct FitParameter
    fixed::Bool = false
    value::Float64 = 0.0
    const edit_buf::Ref{Cdouble} = Ref(0.0)
end

# Per-view fit configuration, kept in one struct so the side-panel fitting UI
# can be driven from it. The fit follows the view's first layer.
@kwdef mutable struct FitSettings
    fit_type::Ref{Cint} = Ref(Cint(0))
    live::Bool = true
    requested::Bool = false
    restrict_x::Bool = false
    x_roi::LinearROI = LinearROI()
    amplitude_sign::Int = 1
    popt::Maybe{Vector{Float64}} = nothing
    retcode::Maybe{Symbol} = nothing
    # Wall time of the most recent fit, in seconds.
    elapsed::Float64 = 0.0
    # Sampled model curve, refreshed by compute_fit! on each successful fit so
    # the GUI can overlay it without re-evaluating per frame.
    const model_x::Vector{Float64} = Float64[]
    const model_y::Vector{Float64} = Float64[]
    # Per-parameter fix flags + values for the current fit type. Rebuilt when
    # fit_type changes; iteration order matches the positional popt layout.
    const params::OrderedDict{String, FitParameter} = OrderedDict{String, FitParameter}()
end

# Parameter names per fit type, matching the order returned by the fit_* funcs.
fit_param_names(name::AbstractString) = if name == "Line"
    ("slope", "intercept")
elseif name == "Gaussian"
    ("y0", "A", "mu", "sigma")
elseif name == "erf"
    ("y0", "A", "center", "fwhm")
elseif name == "sin"
    ("y0", "A", "period", "phi")
else
    ()
end

# Rebuild fit.params for the given fit type, dropping any previous per-param
# state. Called whenever the fit type changes.
function reset_fit_params!(fit::FitSettings)
    empty!(fit.params)
    for pname in fit_param_names(FIT_TYPES[fit.fit_type[] + 1])
        fit.params[pname] = FitParameter()
    end
end

# Collapse fit.params into the Vector{Maybe{Float64}} layout that the fit_*
# functions consume. Returns `nothing` when no slot is pinned so the solver
# takes its fast path.
function fixed_vector(fit::FitSettings)
    if !any(p.fixed for p in values(fit.params))
        return nothing
    end
    return [p.fixed ? p.value : nothing for p in values(fit.params)]
end

function fit_wanted(fit::FitSettings, data_updated::Bool)
    requested = fit.requested
    fit.requested = false
    return requested || (fit.live && data_updated)
end

# Re-run the selected fit against the plot's current X/Y samples. Called from
# the draw_plot data-update path so the popt stays in sync with what's shown.
function compute_fit!(fit::FitSettings, ydata::AbstractVector,
                      xdata::Maybe{AbstractVector}=nothing;
                      sigma::Maybe{AbstractVector}=nothing)
    if fit.restrict_x && isassigned(fit.x_roi)
        xs = isnothing(xdata) ? eachindex(ydata) : xdata
        lo = fit.x_roi.start
        hi = lo + fit.x_roi.length
        idx = findall(x -> lo <= x <= hi, xs)
        ydata = ydata[idx]
        xdata = xs[idx]
        sigma = isnothing(sigma) ? nothing : sigma[idx]
    end

    name = FIT_TYPES[fit.fit_type[] + 1]
    fixed = fixed_vector(fit)
    t0 = time_ns()
    if name == "Line"
        fit.popt, fit.retcode = fit_line(ydata, xdata; sigma, fixed)
    elseif name == "Gaussian"
        fit.popt, fit.retcode = fit_gaussian(ydata, xdata; sigma, fixed,
                                             A_sign=fit.amplitude_sign)
    elseif name == "erf"
        fit.popt, fit.retcode = fit_erf(ydata, xdata; sigma, fixed)
    elseif name == "sin"
        fit.popt, fit.retcode = fit_sin(ydata, xdata; sigma, fixed)
    else
        fit.popt = nothing
        fit.retcode = nothing
    end
    fit.elapsed = (time_ns() - t0) / 1e9

    update_fit_curve!(fit, ydata, xdata)
end

# Sample `model(x)` onto `xs`/`ys` over n evenly spaced points in [xmin, xmax].
function sample_model!(xs::Vector{Float64}, ys::Vector{Float64},
                       model, xmin::Float64, xmax::Float64, n::Int)
    resize!(xs, n)
    resize!(ys, n)
    step = (xmax - xmin) / (n - 1)
    @inbounds for i in 1:n
        x = xmin + (i - 1) * step
        xs[i] = x
        ys[i] = model(x)
    end
end

# Refresh fit.model_x/fit.model_y from the current popt so the GUI can overlay
# the fitted curve without re-evaluating per frame. Clears the buffers if
# there's no popt or the X range is degenerate.
function update_fit_curve!(fit::FitSettings, ydata::AbstractVector,
                           xdata::Maybe{AbstractVector})
    if isnothing(fit.popt)
        empty!(fit.model_x)
        empty!(fit.model_y)
        return
    end

    xs = isnothing(xdata) ? eachindex(ydata) : xdata
    xmin, xmax = Float64(minimum(xs)), Float64(maximum(xs))
    if !isfinite(xmin) || !isfinite(xmax) || xmin == xmax
        empty!(fit.model_x)
        empty!(fit.model_y)
        return
    end

    p = fit.popt
    name = FIT_TYPES[fit.fit_type[] + 1]
    model = if name == "Line"
        x -> p[1] * x + p[2]
    elseif name == "Gaussian"
        x -> gaussian(x, p[1], p[2], p[3], p[4])
    elseif name == "erf"
        x -> erf(x, p[1], p[2], p[3], p[4])
    elseif name == "sin"
        x -> sinusoid(x, p[1], p[2], p[3], p[4])
    end
    sample_model!(fit.model_x, fit.model_y, model, xmin, xmax, 200)
end

# Overlay the fitted model curve on the current ImPlot plot, if any, and the
# draggable X restriction band when enabled. Moving the band requests a refit.
function draw_fit_overlay(view_id, fit::FitSettings)
    name = FIT_TYPES[fit.fit_type[] + 1]
    if !isempty(fit.model_x)
        ImPlot.PlotLine("$(name) fit", fit.model_x, fit.model_y)
    end

    if name != "None" && fit.restrict_x
        if !isassigned(fit.x_roi)
            limits = ImPlot.GetPlotLimits()
            fit.x_roi = default_roi(fit.x_roi, limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max, 1)
            fit.requested = true
        end
        col = ROI_COLORS[3]
        new_roi = drag_roi(fit.x_roi, view_id, "fit-restrict-x",
                           ImVec4(col.x, col.y, col.z, 0.75), Ref(false))
        if !isnothing(new_roi) && new_roi != fit.x_roi
            fit.x_roi = new_roi
            fit.requested = true
        end
    end
end

# --- Plot type payloads ---
#
# A `PlotType` is what a SpecView's `prepare!` hands back each frame, drawn by
# `plot_frame!`.

abstract type PlotType end

# 1D series. `style` selects the ImPlot primitive:
#   :line    → PlotLine
#   :scatter → PlotScatter
struct Line <: PlotType
    xs
    ys
    label::String
    style::Symbol
    # Explicit per-series color, used when a color channel groups a layer into
    # series. nothing lets ImPlot cycle its palette as usual.
    color::Maybe{ig.ImVec4}
    opacity::Float64
end

# Bar series, for histograms and any vector drawn as bars.
struct Bars <: PlotType
    xs
    ys
    label::String
    bar_size::Float64
end

# Shaded band + central line, sharing one legend entry. Used for binned
# correlations where `lower`/`upper` bound the spread around `line_ys`.
struct Band <: PlotType
    xs
    lower
    upper
    line_ys
    label::String
end

# Colormapped 2D data, already rendered into `gpu`'s texture. `x_axis`/`y_axis`
# may be nothing (defaults to pixel coords).
struct Image <: PlotType
    data
    x_axis::Maybe{AbstractVector}
    y_axis::Maybe{AbstractVector}
    gpu::GPUHeatmap
end

# Nothing to draw this frame. `message`, when non-empty, is shown in place of
# the plot.
struct Empty <: PlotType
    message::String
end

# Colorbar interaction state. `clip_min`/`clip_max` are the values fed to
# the colormap shader; `display_min`/`display_max` are the visible range
# shown on the colorbar axis (>= clip range, controlled by mouse wheel).
@kwdef mutable struct ColorbarState
    const autoscale::Ref{Bool} = Ref(true)
    const clip_min::Ref{Cdouble} = Ref(0.0)
    const clip_max::Ref{Cdouble} = Ref(1.0)
    const display_min::Ref{Cdouble} = Ref(0.0)
    const display_max::Ref{Cdouble} = Ref(1.0)
    drag::Symbol = :none
    display_zoomed::Bool = false
end

# Matrix-rendering state: GPU heatmap resources, colormap log toggle, colorbar
# state, ROI overlay bookkeeping. Lives on Plot whenever the plotted data is a
# matrix.
@kwdef mutable struct ImageState
    const fixed_aspect::Ref{Bool} = Ref(true)
    const log_scale::Ref{Bool} = Ref(false)
    colormap::Cint = turbo_colormap()
    const colorbar::ColorbarState = ColorbarState()
    gpu_heatmap::Union{Nothing, GPUHeatmap} = nothing
end

# Pairs samples from two VariableData stores on matching train IDs. Owns the
# paired history buffers and an optional binning accumulator. Pure data
# plumbing — no ImGui state.
@kwdef mutable struct VariableTrainmatcher
    const x_data::Vector{Float64} = Float64[]
    const y_data::Vector{Float64} = Float64[]
    accu::Maybe{Scalar1dScan} = nothing
    # Last vector-mode tid consumed, so we only copy once per matched train.
    last_vector_tid::Int = -1
end

# Walk updated_variables for x_name/y_name and append pairs for any tid present
# in both x.scalar_tids and y.scalar_tids. Routes through accu if active.
# Returns true if any pair was appended.
function ingest_scalar!(m::VariableTrainmatcher, x_store, y_store,
                        updated_variables, x_name, y_name)
    if !haskey(updated_variables, x_name) && !haskey(updated_variables, y_name)
        return false
    end
    new_tids = union(get(updated_variables, x_name, Set{Int}()),
                     get(updated_variables, y_name, Set{Int}()))

    appended = false
    for tid in new_tids
        xi = findfirst(==(tid), x_store.scalar_tids)
        yi = findfirst(==(tid), y_store.scalar_tids)
        if !isnothing(xi) && !isnothing(yi)
            xv = x_store.data[xi]
            yv = y_store.data[yi]
            if !isfinite(xv) || !isfinite(yv)
                continue
            end
            push!(m.x_data, xv)
            push!(m.y_data, yv)
            if !isnothing(m.accu)
                append!(m.accu, xv, yv)
            end
            appended = true
        end
    end
    return appended
end

# Copy both vector buffers when both stores share a fresh trainId. Returns
# true if a copy happened.
function ingest_vector!(m::VariableTrainmatcher, x_store, y_store)
    if !(x_store.data isa AbstractVector) || !(y_store.data isa AbstractVector)
        return false
    end
    if x_store.trainId != y_store.trainId || x_store.trainId == m.last_vector_tid
        return false
    end
    resize!(m.x_data, length(x_store.data))
    resize!(m.y_data, length(y_store.data))
    copyto!(m.x_data, x_store.data)
    copyto!(m.y_data, y_store.data)
    m.last_vector_tid = x_store.trainId
    return true
end

# Reconcile accu with the requested resolution; rebuild from raw history when
# the resolution changes (covers initial creation, widget edits, swaps, var
# changes). Returns true if the binned series changed.
function set_resolution!(m::VariableTrainmatcher, res::Cfloat)
    if res > 0 && (isnothing(m.accu) || m.accu.axes[1].resolution != res)
        m.accu = Scalar1dScan(Float64(res))
        for i in eachindex(m.x_data, m.y_data)
            append!(m.accu, m.x_data[i], m.y_data[i])
        end
        return true
    elseif res <= 0 && !isnothing(m.accu)
        m.accu = nothing
        return true
    end
    return false
end

function Base.empty!(m::VariableTrainmatcher)
    empty!(m.x_data)
    empty!(m.y_data)
    m.accu = nothing
    m.last_vector_tid = -1
end

# In-place x↔y swap; the accu is rebuilt on the next set_resolution! call.
function swap!(m::VariableTrainmatcher)
    for i in eachindex(m.x_data, m.y_data)
        m.x_data[i], m.y_data[i] = m.y_data[i], m.x_data[i]
    end
    m.accu = nothing
end

# Where a SpecView's spec comes from, see refresh_spec!.
abstract type SpecSource end

# The default plot of a variable, with model curves drawn over it if any.
struct DefaultSpec <: SpecSource
    variable::String
    models::Vector{ModelOverlay}
end

# A correlation of two variables picked in the plot window, authored as a
# lookup spec (see correlation_spec).
@kwdef struct CorrelationSpec <: SpecSource
    # The X and Y variables, "" until there's one to pick.
    selected::Vector{String} = ["", ""]
    # Refreshed each frame from client.variable_data; used by the X/Y combos.
    variable_names::Vector{String} = String[]
end

# A plot that `variable` advertises under `name`, besides its default one.
struct AdvertisedSpec <: SpecSource
    variable::String
    name::String
end

# One layer of a SpecView. `image` is the state of a rect layer, `matcher` pairs
# the two variables of a lookup layer, and `series` holds the series a color
# channel groups the data into, rebuilt when it updates.
@kwdef mutable struct ViewLayer
    const spec::LayerSpec
    image::Maybe{ImageState} = nothing
    const matcher::Maybe{VariableTrainmatcher} = nothing
    const series::Vector{PlotType} = PlotType[]
end

# The curves of one of a spec's models, resampled when its parameters update.
@kwdef struct ViewModel
    overlay::ModelOverlay
    xs::Vector{Float64} = Float64[]
    curves::Vector{PlotType} = PlotType[]
end

# Renders a PlotSpec from its source. Synthesised specs are rebuilt whenever
# the source's `spec_key` changes.
@kwdef mutable struct SpecView
    const source::SpecSource
    const id::String
    const fit::FitSettings = FitSettings()
    spec_key::Any = nothing
    spec::Maybe{PlotSpec} = nothing
    const layers::Vector{ViewLayer} = ViewLayer[]
    const models::Vector{ViewModel} = ViewModel[]
    const subscribed::Set{String} = Set{String}()
    # Bin width of the trainId lookup layers, 0 for a plain scatter. Follows the
    # X variable's hint until the user touches it.
    const binning_resolution::Ref{Cfloat} = Ref(Cfloat(0))
    resolution_touched::Bool = false
    # ROI parameter values updated locally during a drag, keyed by parameter
    # name. Flushed to the engine when the user releases the mouse so we don't
    # flood it with per-frame updates.
    const pending_roi_updates::Dict{String, AbstractROI} = Dict{String, AbstractROI}()
end

# A plot window: the view, and the state of the plot's axes. `id` is also the
# `##` suffix of the view's widgets.
@kwdef mutable struct Plot
    const id::String
    const view::SpecView
    const open::Ref{Bool} = Ref(true)
    const autoscale_x::Ref{Bool} = Ref(true)
    const autoscale_y::Ref{Bool} = Ref(true)
    const log_x::Ref{Bool} = Ref(false)
    const log_y::Ref{Bool} = Ref(false)
    const show_side_panel::Ref{Bool} = Ref(false)
    dock_id::UInt32 = 0
end

# The default plot of variable `name`, with `models` drawn over it.
function variable_plot(name::AbstractString, counter::Integer, models = ModelOverlay[])
    id = "$(name)##plot-$(counter)"
    view = SpecView(; source = DefaultSpec(String(name), models), id)
    # Its spec can only be synthesised once there's data
    subscribe_variable(state[], name)
    push!(view.subscribed, name)
    return Plot(; id, view)
end

# A plot correlating two variables picked in its window.
function correlation_plot(counter::Integer)
    id = "CorrelationPlot##plot-$(counter)"
    Plot(; id, view = SpecView(; source = CorrelationSpec(), id))
end

# A plot of the spec `variable` advertises as `name`.
function spec_plot(variable::AbstractString, name::AbstractString, counter::Integer)
    id = "$(variable)/$(name)##plot-$(counter)"
    Plot(; id, view = SpecView(; source = AdvertisedSpec(String(variable), String(name)), id))
end

Base.close(plot::Plot) = close(plot.view)

function Base.close(image::ImageState)
    if !isnothing(image.gpu_heatmap)
        destroy!(image.gpu_heatmap)
        image.gpu_heatmap = nothing
    end
end

function Base.close(view::SpecView)
    for layer in view.layers
        if !isnothing(layer.image)
            close(layer.image)
        end
    end
    for name in view.subscribed
        unsubscribe_variable(state[], name)
    end
    empty!(view.subscribed)
end

clear_plot(plot::Plot) = clear_paired_data!(plot.view)

# Drop the history paired by the lookup layers.
function clear_paired_data!(view::SpecView)
    for layer in view.layers
        if !isnothing(layer.matcher)
            empty!(layer.matcher)
        end
    end
end

function check_plot_interaction!(plot)
    io = ig.GetIO()
    mouse_wheel = unsafe_load(io.MouseWheel)
    # Disable autoscale during the drag (not on release) so ImPlot's box zoom,
    # which commits on release, isn't overridden by apply_autoscale that frame.
    dragging = ig.IsMouseDragging(ig.ImGuiMouseButton_Left) ||
               ig.IsMouseDragging(ig.ImGuiMouseButton_Right)
    interacting = dragging || mouse_wheel != 0

    x_hovered = ImPlot.IsAxisHovered(ImPlot.ImAxis_X1)
    y_hovered = ImPlot.IsAxisHovered(ImPlot.ImAxis_Y1)
    plot_hovered = ImPlot.IsPlotHovered()

    # Disable autoscale on the axes being interacted with
    if interacting
        if plot_hovered
            plot.autoscale_x[] = false
            plot.autoscale_y[] = false
        elseif x_hovered
            plot.autoscale_x[] = false
        elseif y_hovered
            plot.autoscale_y[] = false
        end
    end

    # Double-click to re-enable autoscale
    if ig.IsMouseDoubleClicked(ig.ImGuiMouseButton_Left)
        if plot_hovered
            plot.autoscale_x[] = true
            plot.autoscale_y[] = true
        elseif x_hovered
            plot.autoscale_x[] = true
        elseif y_hovered
            plot.autoscale_y[] = true
        end
    end
end

"""Draw a small toggle button that appears highlighted when active."""
function toggle_button(label, active::Bool)
    if active
        ig.PushStyleColor(ig.ImGuiCol_Button, unsafe_load(ig.GetStyleColorVec4(ig.ImGuiCol_ButtonActive)))
    end
    clicked = ig.SmallButton(label)
    if active
        ig.PopStyleColor()
    end
    return clicked
end

"""Draw the autoscale toggle button group: [X] [Y] [XY]"""
function autoscale_buttons(plot)
    ig.AlignTextToFramePadding()
    ig.Text("Autoscale:")
    ig.SameLine()
    if toggle_button("X##$(plot.id)", plot.autoscale_x[])
        plot.autoscale_x[] = !plot.autoscale_x[]
    end
    ig.SameLine()
    if toggle_button("Y##$(plot.id)", plot.autoscale_y[])
        plot.autoscale_y[] = !plot.autoscale_y[]
    end
    ig.SameLine()
    both = plot.autoscale_x[] && plot.autoscale_y[]
    if toggle_button("XY##$(plot.id)", both)
        new_state = !both
        plot.autoscale_x[] = new_state
        plot.autoscale_y[] = new_state
    end
end

"""Apply log10 scale to X/Y axes based on plot state. Call after BeginPlot,
before any plotting calls."""
function apply_log_scales(plot)
    if plot.log_x[]
        ImPlot.SetupAxisScale(ImPlot.ImAxis_X1, ImPlot.ImPlotScale_Log10)
    end
    if plot.log_y[]
        ImPlot.SetupAxisScale(ImPlot.ImAxis_Y1, ImPlot.ImPlotScale_Log10)
    end
end

"""Draw the log-scale toggle button group: [logX] [logY]"""
function log_scale_buttons(plot)
    ig.AlignTextToFramePadding()
    ig.Text("Log:")
    ig.SameLine()
    if toggle_button("X##log-$(plot.id)", plot.log_x[])
        plot.log_x[] = !plot.log_x[]
    end
    ig.SameLine()
    if toggle_button("Y##log-$(plot.id)", plot.log_y[])
        plot.log_y[] = !plot.log_y[]
    end
end

"""Call per-axis SetNextAxisToFit based on autoscale state."""
function apply_autoscale(plot)
    if plot.autoscale_x[] && plot.autoscale_y[]
        ImPlot.SetNextAxesToFit()
    elseif plot.autoscale_x[]
        ImPlot.SetNextAxisToFit(ImPlot.ImAxis_X1)
    elseif plot.autoscale_y[]
        ImPlot.SetNextAxisToFit(ImPlot.ImAxis_Y1)
    end
end

# zfp accuracy `k` input (0 = lossless), ratio, and throughput readout. `id`
# namespaces the imgui widgets, `name` is the qualified variable name to retune.
function draw_compression_settings(id, name, k::Ref{Cfloat}, store)
    compressed = isfinite(store.compression_ratio)
    enabled = compressed && store.compress
    if !enabled
        ig.BeginDisabled()
    end
    ig.SetNextItemWidth(120)
    if ig.InputFloat("zfp accuracy k##$(id)", k, 0.1f0, 1.0f0, "%.2f")
        set_subscription_k(state[], name, Float64(k[]))
    end
    ig.SameLine()
    ig.TextDisabled("(0 = lossless)")
    if !enabled
        ig.EndDisabled()
    end
    if !store.compress
        ig.TextDisabled("Compression disabled for this variable")
    elseif compressed
        ig.TextDisabled(@sprintf("zfp: %.1fx", store.compression_ratio))
    elseif store.received_bytes > 0
        ig.TextDisabled("Variable is not compressed")
    end
    if store.received_bytes > 0
        ig.TextDisabled(@sprintf("%.1f MB/s @ 10Hz", store.received_bytes * 10 / 1e6))
    end
end

# Interactive colorbar. Draws ImPlot.ColormapScale spanning the display range
# and overlays two horizontal handles at clip_min/clip_max. Returns true when
# the clip range changed and the colormap output needs re-rendering.
#
# Hovering: drag a handle to set clip_min/clip_max (disables colorbar
# autoscale); mouse wheel zooms the display range around the cursor.
function interactive_colorbar(img::ImageState, id, size::ImVec2)
    cb = img.colorbar
    display_min = cb.display_min[]
    display_max = cb.display_max[]
    clip_min = cb.clip_min[]
    clip_max = cb.clip_max[]
    # In log mode all four refs hold log10(value); the colorbar renders that
    # space directly and tick labels read as exponents.
    tick_format = img.log_scale[] ? "1e%g" : "%g"

    # ColormapScale itself does not consume mouse input — without an overlay
    # button, clicks fall through to the parent window and start a window
    # move. Mark it as overlap-allowed and stack an InvisibleButton on top to
    # capture clicks/drags for the handles.
    ig.SetNextItemAllowOverlap()
    start_pos = ig.GetCursorScreenPos()
    ImPlot.ColormapScale("##colorbar_$(id)",
                         display_min, display_max,
                         size, tick_format,
                         ImPlot.ImPlotColormapScaleFlags_None,
                         img.colormap)
    rect_min = ig.GetItemRectMin()
    rect_max = ig.GetItemRectMax()

    ig.SetCursorScreenPos(start_pos)
    ig.InvisibleButton("##colorbar_input_$(id)",
                       ImVec2(rect_max.x - rect_min.x, rect_max.y - rect_min.y))
    hovered = ig.IsItemHovered()
    active = ig.IsItemActive()

    # ColormapScale insets the gradient bar by PlotPadding inside its frame
    pad_y = unsafe_load(ImPlot.GetStyle().PlotPadding).y
    bar_top = rect_min.y + pad_y
    bar_bot = rect_max.y - pad_y
    bar_h = max(bar_bot - bar_top, 1.0f0)
    span = display_max - display_min
    safe_span = span == 0 ? 1.0 : span

    value_to_y(v) = bar_bot - Float32(clamp((v - display_min) / safe_span, 0.0, 1.0)) * bar_h
    y_to_value(y) = display_min + clamp((bar_bot - y) / bar_h, 0.0f0, 1.0f0) * safe_span

    y_min_px = value_to_y(clip_min)
    y_max_px = value_to_y(clip_max)

    # Highlight the handle nearest the cursor while hovered/active
    threshold = 8.0f0
    near_handle = :none
    if hovered || active
        mouse_y = ig.GetMousePos().y
        d_min = abs(mouse_y - y_min_px)
        d_max = abs(mouse_y - y_max_px)
        if active && cb.drag !== :none
            near_handle = cb.drag
        elseif d_min <= d_max && d_min < threshold
            near_handle = :min
        elseif d_max < threshold
            near_handle = :max
        end
    end

    draw = ig.GetWindowDrawList()
    base_color = ig.GetColorU32(ig.ImGuiCol_Text, 0.5f0)
    hover_color = ig.GetColorU32(ImVec4(1.0f0, 0.2f0, 0.2f0, 0.5f0))
    thickness = 5.0f0
    min_color = near_handle === :min ? hover_color : base_color
    max_color = near_handle === :max ? hover_color : base_color
    ig.AddLine(draw, ImVec2(rect_min.x, y_min_px), ImVec2(rect_max.x, y_min_px), min_color, thickness)
    ig.AddLine(draw, ImVec2(rect_min.x, y_max_px), ImVec2(rect_max.x, y_max_px), max_color, thickness)

    changed = false

    if ig.IsItemActivated()
        mouse_y = ig.GetMousePos().y
        d_min = abs(mouse_y - y_min_px)
        d_max = abs(mouse_y - y_max_px)
        cb.drag = d_min <= d_max ? :min : :max
    end

    if active && cb.drag !== :none
        mouse_y = ig.GetMousePos().y
        new_v = y_to_value(mouse_y)
        eps = 1e-9 * max(abs(safe_span), 1.0)
        if cb.drag === :min
            cb.clip_min[] = min(new_v, cb.clip_max[] - eps)
        else
            cb.clip_max[] = max(new_v, cb.clip_min[] + eps)
        end
        cb.autoscale[] = false
        changed = true
    elseif !active
        cb.drag = :none
    end

    if hovered && !active
        wheel = unsafe_load(ig.GetIO().MouseWheel)
        if wheel != 0
            mouse_y = ig.GetMousePos().y
            anchor = y_to_value(mouse_y)
            factor = wheel > 0 ? 0.85 : 1 / 0.85
            cb.display_min[] = anchor + (display_min - anchor) * factor
            cb.display_max[] = anchor + (display_max - anchor) * factor
            cb.display_zoomed = true
        end
    end

    return changed
end

# Color palette cycled through when a plot has multiple ROI overlays.
const ROI_COLORS = ImVec4[
    ImVec4(1.00, 1.00, 1.00, 1.0),  # white
    ImVec4(1.00, 0.20, 0.40, 1.0),  # red
    ImVec4(1.00, 0.55, 0.20, 1.0),  # orange
    ImVec4(1.00, 0.30, 0.85, 1.0),  # magenta
    ImVec4(1.00, 0.65, 0.75, 1.0),  # salmon pink
    ImVec4(0.75, 0.20, 1.00, 1.0),  # violet
    ImVec4(0.85, 0.10, 0.10, 1.0),  # crimson
    ImVec4(0.95, 0.75, 0.60, 1.0),  # peach
    ImVec4(0.60, 0.30, 0.20, 1.0),  # brown
    ImVec4(0.20, 0.20, 0.20, 1.0),  # near-black
]

# Default extent for an unassigned ROI along one axis: the middle half of
# [lo, hi], shifted by `idx` so successive ROIs don't fully overlap.
function default_roi_span(lo, hi, idx)
    span = hi - lo
    (lo + span / 4 + span * 0.05 * (idx - 1), span / 2)
end

function default_roi(::RectROI, x_min, x_max, y_min, y_max, idx)
    x, w = default_roi_span(x_min, x_max, idx)
    y, h = default_roi_span(y_min, y_max, idx)
    RectROI(x, y, w, h)
end

function default_roi(roi::LinearROI, x_min, x_max, y_min, y_max, idx)
    lo, hi = roi.axis == :x ? (x_min, x_max) : (y_min, y_max)
    LinearROI(default_roi_span(lo, hi, idx)...; axis=roi.axis)
end

# Corners (x1, y1, x2, y2) of the DragRect drawn for an ROI. A LinearROI is
# pinned to the current plot limits along its other axis.
roi_corners(roi::RectROI) = (roi.corner_x, roi.corner_y, roi.corner_x + roi.width, roi.corner_y + roi.height)
function roi_corners(roi::LinearROI)
    limits = ImPlot.GetPlotLimits()
    if roi.axis == :x
        (roi.start, limits.Y.Min, roi.start + roi.length, limits.Y.Max)
    else
        (limits.X.Min, roi.start, limits.X.Max, roi.start + roi.length)
    end
end

# Draw the drag handles for an ROI in colour `col`, setting `held` while any of
# them is being dragged. Returns the updated ROI, or nothing if it didn't move.
function drag_roi(roi::RectROI, view_id, param_name, col, held)
    x1, y1, x2, y2 = map(v -> Ref(Cdouble(v)), roi_corners(roi))
    id = int32_hash(view_id, param_name)
    if ImPlot.DragRect(id, x1, y1, x2, y2, col, 0, C_NULL, C_NULL, held)
        xlo, xhi = minmax(x1[], x2[])
        ylo, yhi = minmax(y1[], y2[])
        RectROI(xlo, ylo, xhi - xlo, yhi - ylo)
    else
        nothing
    end
end

# Like matplotlib's axvspan/axhspan: a shaded band spanning the plot with a
# draggable line on each edge and a point in the centre that moves the whole
# band. NoFit keeps the handles from stretching an auto-fitted axis.
function drag_roi(roi::LinearROI, view_id, param_name, col, held)
    x1, y1, x2, y2 = roi_corners(roi)
    p1 = ImPlot.PlotToPixels(x1, y1)
    p2 = ImPlot.PlotToPixels(x2, y2)
    fill_col = ig.GetColorU32(ImVec4(col.x, col.y, col.z, col.w / 4))
    ImPlot.PushPlotClipRect()
    ig.AddRectFilled(ImPlot.GetPlotDrawList(),
                     ImVec2(min(p1.x, p2.x), min(p1.y, p2.y)),
                     ImVec2(max(p1.x, p2.x), max(p1.y, p2.y)), fill_col)
    ImPlot.PopPlotClipRect()

    drag_line = roi.axis == :x ? ImPlot.DragLineX : ImPlot.DragLineY
    lo = Ref(Cdouble(roi.start))
    hi = Ref(Cdouble(roi.start + roi.length))
    held_hi = Ref(false)
    flags = ImPlot.ImPlotDragToolFlags_NoFit
    moved_lo = drag_line(int32_hash(view_id, param_name), lo, col, 1, flags, C_NULL, C_NULL, held)
    moved_hi = drag_line(int32_hash(view_id, param_name * ".hi"), hi, col, 1, flags, C_NULL, C_NULL, held_hi)

    # Centre handle, like ImPlot's DragPoint but following the mouse only along
    # the ROI's axis so the band can't be dragged sideways.
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    center = ImPlot.PlotToPixels(cx, cy)
    grab = 4
    center_id = ig.GetID(int32_hash(view_id, param_name * ".center"))
    ig.igKeepAliveID(center_id)
    bb = ig.ImRect(ImVec2(center.x - grab, center.y - grab), ImVec2(center.x + grab, center.y + grab))
    hovered_center = Ref(false)
    held_center = Ref(false)
    ig.igButtonBehavior(bb, center_id, hovered_center, held_center, 0)
    if hovered_center[] || held_center[]
        ig.SetMouseCursor(ig.ImGuiMouseCursor_Hand)
    end
    moved_center = held_center[] && ig.IsMouseDragging(ig.ImGuiMouseButton_Left)
    if moved_center
        mouse = ImPlot.GetPlotMousePos()
        shift = roi.axis == :x ? mouse.x - cx : mouse.y - cy
        lo[] += shift
        hi[] += shift
        center = roi.axis == :x ? ImPlot.PlotToPixels(mouse.x, cy) : ImPlot.PlotToPixels(cx, mouse.y)
    end
    ImPlot.PushPlotClipRect()
    ig.AddCircleFilled(ImPlot.GetPlotDrawList(), center, grab, ig.GetColorU32(col))
    ImPlot.PopPlotClipRect()

    if held_hi[] || held_center[]
        held[] = true
    end
    if moved_lo || moved_hi || moved_center
        a, b = minmax(lo[], hi[])
        LinearROI(a, b - a; axis=roi.axis)
    else
        nothing
    end
end

# Data extent used to seed unassigned ROIs.
roi_bounds(frame::Image) = image_bounds(frame)[3:end]
function roi_bounds(::PlotType)
    limits = ImPlot.GetPlotLimits()
    (limits.X.Min, limits.X.Max, limits.Y.Min, limits.Y.Max)
end

# Drawn once per view, after its frames: the fit curve and the spec's ROIs. An
# ROI edits the parameter it's named after, which also holds its live value; the
# spec only gives the initial extent of an unassigned one.
function draw_view_overlays(view::SpecView, frames)
    i = findfirst(f -> !(f isa Empty), frames)
    if any(f -> f isa Union{Line, Bars, Band}, frames)
        draw_fit_overlay(view.id, view.fit)
    end
    if isnothing(i)
        return
    end
    x_min, x_max, y_min, y_max = roi_bounds(frames[i])
    client = state[].client
    pending = view.pending_roi_updates
    for (idx, roi_param) in enumerate(view.spec.rois)
        param_name = roi_param.name
        param = get(client.context.parameters, param_name, nothing)
        if isnothing(param) || !(param.value isa AbstractROI)
            continue
        end
        roi = param.value
        if !isassigned(roi)
            if isassigned(roi_param.initial) && typeof(roi_param.initial) == typeof(roi)
                roi = roi_param.initial
            else
                roi = default_roi(roi, x_min, x_max, y_min, y_max, idx)
            end
        end
        col = ROI_COLORS[mod1(idx, length(ROI_COLORS))]
        held = Ref(false)
        new_roi = drag_roi(roi, view.id, param_name,
                           ImVec4(col.x, col.y, col.z, 0.75), held)
        if !isnothing(new_roi)
            roi = new_roi
            if new_roi != param.value
                param.value = new_roi
                pending[param_name] = new_roi
            end
        end

        if !held[] && haskey(pending, param_name)
            client.pending_source_edit = param_name
            change_parameter(Parameter(param_name, pending[param_name]))
            delete!(pending, param_name)
        end

        # Label at the top-left corner in screen space (the Y axis may be
        # inverted, so find it in pixels): above the ROI when there's room,
        # otherwise just inside it, e.g. for a band pinned to the plot's top
        # edge. PlotText centers on its anchor, so offset by half the text size.
        base_size = unsafe_load(ig.GetStyle()).FontSizeBase
        ig.PushFont(C_NULL, base_size * 2)
        text_size = ig.CalcTextSize(param_name)
        x1, y1, x2, y2 = roi_corners(roi)
        p1 = ImPlot.PlotToPixels(x1, y1)
        p2 = ImPlot.PlotToPixels(x2, y2)
        top_left = ImVec2(min(p1.x, p2.x), min(p1.y, p2.y))
        dy = if top_left.y - text_size.y - 2 < ImPlot.GetPlotPos().y
            text_size.y / 2 + 2
        else
            -text_size.y / 2 - 2
        end
        anchor = ImPlot.PixelsToPlot(top_left)
        ImPlot.PushStyleColor(ImPlot.ImPlotCol_InlayText, col)
        # NoFit, otherwise a label pinned to the plot's edge holds back autoscaling
        ImPlot.PlotText(param_name, anchor.x, anchor.y, ImVec2(text_size.x / 2, dy),
                        ImPlot.ImPlotSpec(; Flags=Cint(ImPlot.ImPlotItemFlags_NoFit)))
        ImPlot.PopStyleColor()
        ig.PopFont()
    end
end

# A thin full-height button on the right edge of the plot area that toggles the
# side panel. Laid out as a normal item (after the plot/colorbar) so it neither
# overlaps nor leaks clicks to the colorbar.
function side_panel_tab(plot, tab_w, height)
    ig.SameLine()
    glyph = plot.show_side_panel[] ? ">" : "<"
    if ig.Button("$(glyph)##sidepanel-tab-$(plot.id)", ImVec2(tab_w, height))
        plot.show_side_panel[] = !plot.show_side_panel[]
    end
end

function draw_fitting_settings(id, fit::FitSettings)
    if ig.CollapsingHeader("Fitting##$(id)")
        ig.SetNextItemWidth(150)
        if ig.Combo("Fit type##$(id)", fit.fit_type, FIT_TYPES, length(FIT_TYPES))
            # Stale popt/retcode would mismatch the new fit type's parameter list.
            fit.popt = nothing
            fit.retcode = nothing
            empty!(fit.model_x)
            empty!(fit.model_y)
            reset_fit_params!(fit)
        end

        name = FIT_TYPES[fit.fit_type[] + 1]
        if name == "None"
            return
        end

        if @c ig.Checkbox("Live fitting##$(id)", &fit.live)
            fit.requested = fit.live
        end
        ig.SameLine()
        ig.BeginDisabled(fit.live)
        if ig.Button("Fit##$(id)")
            fit.requested = true
        end
        ig.EndDisabled()

        if name == "Gaussian"
            ig.AlignTextToFramePadding()
            ig.Text("Amplitude is:")
            ig.SameLine()
            if toggle_button("positive##a-sign-$(id)", fit.amplitude_sign > 0)
                fit.amplitude_sign = 1
                fit.requested = true
            end
            ig.SameLine()
            if toggle_button("negative##a-sign-$(id)", fit.amplitude_sign < 0)
                fit.amplitude_sign = -1
                fit.requested = true
            end
        end

        checkbox_x = ig.GetCursorPosX()
        if @c ig.Checkbox("##restrict-x-$(id)", &fit.restrict_x)
            fit.requested = true
        end
        ig.SetItemTooltip("Only fit the samples inside a draggable range of the X axis")
        ig.SameLine()
        header_indent = ig.GetCursorPosX() - checkbox_x
        ig.BeginDisabled(!fit.restrict_x)
        expanded = ig.CollapsingHeader("Restrict X##$(id)")
        ig.EndDisabled()

        if expanded && fit.restrict_x && isassigned(fit.x_roi)
            ig.Indent(header_indent)
            lo = Ref(Cdouble(fit.x_roi.start))
            hi = Ref(Cdouble(fit.x_roi.start + fit.x_roi.length))
            speed = Cfloat(fit.x_roi.length / 100)
            ig.SetNextItemWidth(120)
            edited = ig.DragScalar("Min##$(id)", ig.ImGuiDataType_Double, lo, speed,
                                   C_NULL, C_NULL, "%.6e")
            ig.SetNextItemWidth(120)
            if ig.DragScalar("Max##$(id)", ig.ImGuiDataType_Double, hi, speed,
                             C_NULL, C_NULL, "%.6e")
                edited = true
            end

            if edited
                a, b = minmax(lo[], hi[])
                fit.x_roi = LinearROI(a, b - a; axis=fit.x_roi.axis)
                fit.requested = true
            end
            ig.Unindent(header_indent)
        end

        ig.Spacing()
        ig.Spacing()
        ig.Text("Parameters:")

        for (i, (pname, param)) in enumerate(fit.params)
            ig.PushID("fit-param-$(id)-$(pname)")
            @c ig.Checkbox("##fix", &param.fixed)
            ig.SetItemTooltip("Fix this parameter to a specific value")
            ig.SameLine()

            # Mirror the fitted value into both the committed value and widget
            # buffer for free slots so the user can flip "fixed" on and edit
            # from the current fit.
            if !param.fixed && !isnothing(fit.popt)
                param.value = fit.popt[i]
                param.edit_buf[] = fit.popt[i]
            end

            flags = param.fixed ? ig.ImGuiInputTextFlags_None :
                                  ig.ImGuiInputTextFlags_ReadOnly
            ig.SetNextItemWidth(130)
            # InputDouble's bound buffer updates per keystroke (only the commit
            # to `param.value` is gated on Enter / focus-loss), so comparing
            # edit_buf to value detects uncommitted changes mid-edit.
            uncommitted = param.fixed && param.edit_buf[] != param.value
            if uncommitted
                ig.PushStyleColor(ig.ImGuiCol_FrameBg, ig.IM_COL32(143, 98, 0, 255))
            end
            ig.InputDouble("$(pname)", param.edit_buf, 0.0, 0.0, "%.8g", flags)
            if uncommitted
                ig.PopStyleColor()
            end
            if param.fixed && ig.IsItemDeactivatedAfterEdit()
                param.value = param.edit_buf[]
            end
            ig.SameLine()
            ig.TextDisabled(param.fixed ? "(fixed)" : "(fitted)")
            ig.PopID()
        end

        if !isnothing(fit.popt)
            ig.TextDisabled(@sprintf("Fit time: %.2f ms", fit.elapsed * 1e3))
        elseif !isnothing(fit.retcode)
            ig.TextWrapped("Fit failed: $(fit.retcode)")
        end
    end
end

# --- SpecView methods ---

# A dim's lookup values as a plain vector, but only when ForwardOrdered; an
# unordered/reverse lookup wouldn't map onto the stretched heatmap axis, so
# fall back to pixel indices (nothing).
function forward_lookup(data::DimArray, dim::Int)
    lo = lookup(data)[dim]

    if !(lo isa DD.Lookup)
        lo isa DenseVector ? lo : collect(lo)
    else
        DD.order(lo) isa DD.ForwardOrdered ? parent(lo) : nothing
    end
end

# Everything a variable's default spec depends on, it's only resynthesised when
# this changes.
function default_spec_key(store)
    data = store.data
    dim_names = data isa DimArray ? DD.name(DD.dims(data)) : ()
    (data isa CircularBuffer, ndims(data), dim_names, store.title, store.xlabel, store.ylabel,
     store.plot_type, store.fixed_aspect)
end

# The spec of a variable's default plot: a scalar history against its train IDs,
# a vector as a line (or pre-binned bars for a histogram), a matrix as an image.
function default_spec(name, store)
    data = store.data
    dim_field(dim, plain) = data isa DimArray ? string(DD.name(DD.dims(data, dim))) : plain

    layer = if data isa AbstractMatrix
        color = ChannelDef("value", FieldType_Quantitative, "value", false, "turbo", false)
        LayerSpec(String(name), Mark_Rect, 1.0, axis_channel(dim_field(2, "col"), store.xlabel),
                  axis_channel(dim_field(1, "row"), store.ylabel), color, nothing)
    else
        histogram = store.plot_type == :histogram
        x_field = data isa CircularBuffer ? "trainId" : dim_field(1, "index")
        LayerSpec(String(name), histogram ? Mark_Bar : Mark_Line, 1.0,
                  axis_channel(x_field, store.xlabel, histogram), axis_channel("value", store.ylabel),
                  nothing, nothing)
    end
    PlotSpec(name, [layer]; title = store.title, fixed_aspect = store.fixed_aspect)
end

# A quantitative x/y channel. An empty label hides the axis title.
axis_channel(field, label, binned = false) = ChannelDef(field, FieldType_Quantitative, label, false, nothing, binned)

# The spec correlating variable `y_name` against `x_name`: a lookup pulls X's
# values into Y's layer, matched per train for scalar histories and per element
# for vectors.
function correlation_spec(x_name, y_name, x, y)
    key = x.type == VariableType_Scalar ? LookupKey_TrainId : LookupKey_Index
    layer = LayerSpec(String(y_name), Mark_Point, 0.5, axis_channel("x", x.title), axis_channel("value", y.title),
                      nothing, LookupTransform(key, String(x_name), "value", "x"))
    PlotSpec("", [layer])
end

# The number of the dim a field names: a DimArray's dim by name, or any array's
# by position (index, or row/col). nothing if there's no such dim.
function field_dim(data, field)
    if data isa DimArray && DD.hasdim(data, Symbol(field))
        DD.dimnum(data, Symbol(field))
    else
        findfirst(==(field), ndims(data) == 1 ? ("index",) : ("row", "col"))
    end
end

# The values of a field of a 1D variable: `value` is the data itself, `trainId`
# the train IDs of a scalar history, and a dim gives its coordinates.
function field_values(name, store, field)
    data = store.data
    if field == "value"
        if data isa CircularBuffer
            store.scalar_data_cache
        elseif data isa DimArray
            parent(data)
        else
            data
        end
    elseif field == "trainId" && data isa CircularBuffer
        store.scalar_tids_cache
    elseif !(data isa CircularBuffer) && field_dim(data, field) == 1
        dim_values(store, 1)
    else
        throw(SpecError("$(name) has no field \"$(field)\""))
    end
end

# The coordinates along a dim: the variable's explicit axis (`x_axis` runs along
# a matrix's columns), else a DimArray's lookup, else the index range.
function dim_values(store, dim)
    data = store.data
    explicit = ndims(data) == 1 || dim == 2 ? store.x_axis : store.y_axis
    if !isnothing(explicit)
        explicit
    elseif data isa DimArray
        parent(lookup(data)[dim])
    else
        1:size(data, dim)
    end
end

# The ImPlot colormap of a Vega scheme.
function scheme_colormap(scheme)
    colormap = choice(COLOR_SCHEMES, scheme, "scale.scheme")
    isnothing(colormap) ? turbo_colormap() : Cint(colormap)
end

# The colour of series `i` of `n`: sampled along the scheme for a quantitative
# channel (viridis by default), else the i'th of a categorical palette.
function series_color(channel::ChannelDef, i, n)
    if channel.type == FieldType_Quantitative
        # Start at 0.2, the dark end of most schemes vanishes against the background
        t = Cfloat(0.2 + 0.8 * (i - 1) / max(n - 1, 1))
        ImPlot.SampleColormap(t, scheme_colormap(something(channel.scheme, "viridis")))
    elseif isnothing(channel.scheme)
        ImPlot.GetColormapColor(i - 1)
    else
        ImPlot.GetColormapColor(i - 1, scheme_colormap(channel.scheme))
    end
end

# The (x, color) dims of a multi-series layer, whose color channel groups a
# matrix by one dim. An x on `index` runs along each series, i.e. the other dim.
function series_dims(name, data, spec::LayerSpec)
    color_dim = field_dim(data, spec.color.field)
    x_dim = if spec.x.field == "index" && data isa AbstractMatrix && !isnothing(color_dim)
        3 - color_dim
    else
        field_dim(data, spec.x.field)
    end
    if !(data isa AbstractMatrix) || isnothing(x_dim) || isnothing(color_dim) || x_dim == color_dim ||
       spec.y.field != "value"
        throw(SpecError("$(name): a color channel needs a matrix, with x and color on its two dims " *
                        "and y on \"value\""))
    end
    x_dim, color_dim
end

# Rebuild `series` as one frame per coordinate along the color dim. The colours
# are always explicit: ImPlot caches an item's colour and only refreshes it when
# given one, so a changed scheme wouldn't show otherwise.
function group_series!(series, name, store, spec::LayerSpec)
    data = store.data
    x_dim, color_dim = series_dims(name, data, spec)
    xs = dim_values(store, x_dim)
    coords = dim_values(store, color_dim)
    values = data isa DimArray ? parent(data) : data

    empty!(series)
    for (i, ys) in enumerate(eachslice(values; dims = color_dim))
        title = something(spec.color.title, "")
        label = isempty(title) ? string(coords[i]) : "$(title) - $(coords[i])"
        if spec.mark == Mark_Bar
            bar_size = length(xs) > 1 ? Float64(abs(xs[2] - xs[1])) : 1.0
            push!(series, Bars(xs, ys, label, bar_size))
        else
            style = spec.mark == Mark_Point ? :scatter : :line
            push!(series, Line(xs, ys, label, style, series_color(spec.color, i, length(coords)), spec.opacity))
        end
    end
end

# The coordinates of an image axis: the variable's explicit axis, else a
# DimArray's lookup, else nothing for pixel indices.
function image_axis(explicit, data, dim)
    if !isnothing(explicit)
        explicit
    elseif data isa DimArray
        forward_lookup(data, dim)
    else
        nothing
    end
end

# The (x, y) variables a lookup layer pairs.
function lookup_pair(spec::LayerSpec)
    if spec.x.field == spec.lookup.as
        spec.lookup.dataset, spec.data
    else
        spec.data, spec.lookup.dataset
    end
end

# What to draw for a layer pairing two variables: their matched samples, or for
# scalars a band around the binned means once there's a binning resolution.
function lookup_frame(view::SpecView, layer::ViewLayer, updated_variables)
    spec = layer.spec
    x_name, y_name = lookup_pair(spec)
    variable_data = state[].client.variable_data
    if !haskey(variable_data, x_name) || !haskey(variable_data, y_name)
        Empty("Waiting for data: $(x_name), $(y_name)")
    else
        x = variable_data[x_name]
        y = variable_data[y_name]
        by_train = spec.lookup.key == LookupKey_TrainId
        if x.type != y.type || x.type != (by_train ? VariableType_Scalar : VariableType_Vector)
            throw(SpecError("$(y_name): a lookup on $(by_train ? "trainId pairs scalars" : "index pairs vectors"), " *
                            "got a $(var_type_label(x)) and a $(var_type_label(y))"))
        end

        m = layer.matcher
        data_updated = if by_train
            appended = ingest_scalar!(m, x, y, updated_variables, x_name, y_name)
            # Follow the X variable's hint (0 for none) until the user sets a resolution
            if !view.resolution_touched
                view.binning_resolution[] = Cfloat(x.bin_resolution)
            end
            rebinned = set_resolution!(m, view.binning_resolution[])
            appended || rebinned
        else
            ingest_vector!(m, x, y)
        end

        if length(m.x_data) != length(m.y_data)
            Empty("Cannot correlate vectors of different lengths ($(length(m.x_data)) vs $(length(m.y_data))).")
        else
            # Only scalars are ever binned
            binned = !isnothing(m.accu)
            xs = binned ? positions(m.accu, 1) : m.x_data
            ys = binned ? parent(m.accu.mean) : m.y_data
            if layer === view.layers[1] && fit_wanted(view.fit, data_updated)
                compute_fit!(view.fit, ys, xs; sigma = binned ? 1 ./ sqrt.(parent(m.accu.count)) : nothing)
            end

            label = "$(x_name) vs $(y_name)"
            if binned
                half = 0.5 .* parent(m.accu.std)
                Band(xs, ys .- half, ys .+ half, ys, label)
            else
                style = spec.mark == Mark_Point ? :scatter : :line
                Line(xs, ys, label, style, nothing, spec.opacity)
            end
        end
    end
end

# Append what to draw for one layer to `frames`. Throws a SpecError if the spec
# doesn't fit the data. The view's fit follows its first layer.
function layer_frames!(frames, view::SpecView, layer::ViewLayer, updated_variables)
    spec = layer.spec
    name = spec.data
    store = get(state[].client.variable_data, name, nothing)
    if !isnothing(spec.lookup)
        push!(frames, lookup_frame(view, layer, updated_variables))
    elseif isnothing(store) || store.data isa ArrayMetadata
        push!(frames, Empty("Waiting for data: $(name)"))
    elseif !(eltype(store.data) <: Real)
        push!(frames, Empty("$(name): unsupported array type $(typeof(store.data))"))
    elseif isempty(store.data)
        push!(frames, Empty("$(name): array has length 0, nothing to plot"))
    elseif spec.mark == Mark_Rect
        data = store.data
        # Rows run along Y and columns along X
        if !(data isa AbstractMatrix) || field_dim(data, spec.x.field) != 2 ||
           field_dim(data, spec.y.field) != 1
            throw(SpecError("$(name): a rect needs a matrix, with x on its second dim and y on its first"))
        end
        push!(frames, prepare_heatmap!(layer.image, data, image_axis(store.x_axis, data, 2),
                                       image_axis(store.y_axis, data, 1), haskey(updated_variables, name)))
    elseif !isnothing(spec.color)
        if haskey(updated_variables, name) || isempty(layer.series)
            group_series!(layer.series, name, store, spec)
        end
        append!(frames, layer.series)
    elseif !(store.data isa AbstractVector)
        throw(SpecError("$(name): only a rect or a color channel can draw $(ndims(store.data))D data"))
    else
        xs = field_values(name, store, spec.x.field)
        ys = field_values(name, store, spec.y.field)
        if length(xs) != length(ys)
            push!(frames, Empty("$(name): x has $(length(xs)) values but y has $(length(ys))"))
        else
            if layer === view.layers[1] && fit_wanted(view.fit, haskey(updated_variables, name))
                compute_fit!(view.fit, ys, xs)
            end
            if spec.mark == Mark_Bar
                bar_size = length(xs) > 1 ? Float64(abs(xs[2] - xs[1])) : 1.0
                push!(frames, Bars(xs, ys, store.title, bar_size))
            else
                # A lone point would be an invisible line
                style = spec.mark == Mark_Point || length(ys) == 1 ? :scatter : :line
                push!(frames, Line(xs, ys, store.title, style, nothing, spec.opacity))
            end
        end
    end
end

# The function of a model with the parameters `p`, which come in a fixed order.
function model_function(func, name, p)
    if length(p) != 4
        throw(SpecError("$(name): a gaussian takes 4 parameters (y0, A, μ, σ), got $(length(p))"))
    end
    x -> gaussian(x, p[1], p[2], p[3], p[4])
end

# The x extent a model is sampled over: the union of the extents of the layers
# holding train `tid`. A fit is never drawn over another train's data, so layers
# from a different train don't contribute.
function layers_extent(view::SpecView, tid)
    variable_data = state[].client.variable_data
    xmin, xmax = Inf, -Inf
    for layer in view.layers
        name = layer.spec.data
        if !haskey(variable_data, name)
            continue
        end
        store = variable_data[name]
        if store.data isa ArrayMetadata || store.trainId != tid
            continue
        end
        lo, hi = extrema(field_values(name, store, layer.spec.x.field))
        xmin = min(xmin, Float64(lo))
        xmax = max(xmax, Float64(hi))
    end
    return xmin, xmax
end

# Resample a model's curves when its parameters or any of the layers it's drawn
# over update. A matrix of parameters gives one curve per column.
function sample_curves!(view::SpecView, model::ViewModel, updated_variables)
    overlay = model.overlay
    variable_data = state[].client.variable_data
    if haskey(updated_variables, overlay.params) || isempty(model.curves) ||
       any(layer -> haskey(updated_variables, layer.spec.data), view.layers)
        empty!(model.curves)
        if haskey(variable_data, overlay.params)
            params = variable_data[overlay.params]
            if !(params.data isa ArrayMetadata)
                if !(params.data isa AbstractVecOrMat{<:Real})
                    throw(SpecError("$(overlay.params): a model needs a vector or matrix of parameters"))
                end
                xmin, xmax = layers_extent(view, params.trainId)
                values = params.data isa DimArray ? parent(params.data) : params.data
                title = something(overlay.title, overlay.params)

                if isfinite(xmin) && isfinite(xmax) && xmin != xmax
                    for (k, p) in enumerate(eachcol(values))
                        ys = Float64[]
                        sample_model!(model.xs, ys, model_function(overlay.func, overlay.params, p), xmin, xmax, 200)
                        label = size(values, 2) == 1 ? title : "$(title) $(k)"
                        push!(model.curves, Line(model.xs, ys, label, :line, nothing, 1.0))
                    end
                end
            end
        end
    end
end

# The matcher for a lookup layer replacing `previous`. It keeps the history when
# it still pairs the same two variables, swapped over if they traded places.
function carry_matcher(previous::Maybe{ViewLayer}, spec::LayerSpec)
    if isnothing(previous) || isnothing(previous.matcher) ||
       previous.spec.lookup.key != spec.lookup.key
        nothing
    elseif lookup_pair(previous.spec) == lookup_pair(spec)
        previous.matcher
    elseif reverse(lookup_pair(previous.spec)) == lookup_pair(spec)
        swap!(previous.matcher)
        previous.matcher
    else
        nothing
    end
end

# Swap in a new spec. A layer keeps the state of the one it replaces where that
# still applies (the image, the paired history), so they survive e.g. a title
# change.
function set_spec!(view::SpecView, plot::Plot, spec::PlotSpec)
    wanted = datasets(spec)
    for name in setdiff(wanted, view.subscribed)
        # Lossy compression would mangle the fine detail of a matrix drawn as lines
        as_lines = any(layer -> layer.data == name && layer.mark != Mark_Rect && !isnothing(layer.color),
                       spec.layers)
        subscribe_variable(state[], name; k = as_lines ? 0.0 : nothing)
    end
    for name in setdiff(view.subscribed, wanted)
        unsubscribe_variable(state[], name)
    end
    empty!(view.subscribed)
    union!(view.subscribed, wanted)

    old = copy(view.layers)
    empty!(view.layers)
    for (i, layer_spec) in enumerate(spec.layers)
        previous = i <= length(old) ? old[i] : nothing
        image = nothing
        if layer_spec.mark == Mark_Rect
            if isnothing(previous) || isnothing(previous.image)
                # The spec only seeds the toggles
                image = ImageState(; fixed_aspect = Ref(spec.fixed_aspect), log_scale = Ref(layer_spec.color.log))
            else
                image = previous.image
                previous.image = nothing
            end
            # The colormap follows the spec, Vega-Lite's default heatmap scheme is viridis
            image.colormap = scheme_colormap(something(layer_spec.color.scheme, "viridis"))
        end
        matcher = nothing
        if !isnothing(layer_spec.lookup)
            matcher = carry_matcher(previous, layer_spec)
            if isnothing(matcher)
                matcher = VariableTrainmatcher()
                # A new X variable goes back to following its resolution hint
                if isnothing(previous) || isnothing(previous.matcher) ||
                   lookup_pair(previous.spec)[1] != lookup_pair(layer_spec)[1]
                    view.resolution_touched = false
                end
            end
        end
        push!(view.layers, ViewLayer(; spec = layer_spec, image, matcher))
    end
    for layer in old
        if !isnothing(layer.image)
            close(layer.image)
        end
    end

    empty!(view.models)
    for overlay in spec.models
        push!(view.models, ViewModel(; overlay))
    end

    # The spec's scales only seed the log toggles
    if isnothing(view.spec)
        plot.log_x[] = any(layer -> layer.x.log, spec.layers)
        plot.log_y[] = any(layer -> layer.y.log, spec.layers)
    end
    view.spec = spec
end

# The label the data gives a channel whose title the spec leaves open: what the
# variable's default plot shows along it, or the title of a paired variable.
function data_label(layer::LayerSpec, channel::ChannelDef)
    paired = !isnothing(layer.lookup)
    name = paired && channel.field == layer.lookup.as ? layer.lookup.dataset : layer.data
    store = get(state[].client.variable_data, name, nothing)
    if isnothing(store) || store.data isa ArrayMetadata
        ""
    elseif paired
        store.title
    elseif channel.field == "value"
        store.data isa AbstractMatrix ? store.title : store.ylabel
    elseif store.data isa AbstractMatrix
        # xlabel runs along a matrix's columns and ylabel along its rows
        dim = layer.mark == Mark_Rect ? field_dim(store.data, channel.field) : series_dims(name, store.data, layer)[1]
        dim == 2 ? store.xlabel : store.ylabel
    else
        store.xlabel
    end
end

# The label of an axis (:x or :y): the spec's, else the data's for the first
# layer that leaves it open.
function axis_label(spec::PlotSpec, label, axis::Symbol)
    if isnothing(label)
        i = findfirst(layer -> isnothing(getfield(layer, axis).title), spec.layers)
        isnothing(i) ? "" : data_label(spec.layers[i], getfield(spec.layers[i], axis))
    else
        label
    end
end

# The parameters `variable` shows on its plot with @display.
displayed_parameters(variable) = get(state[].client.context.displays, variable, String[])

# `spec` with the @display ROIs of `variable` and `models` added.
function extend_spec(spec::PlotSpec, variable, models)
    parameters = state[].client.context.parameters
    rois = copy(spec.rois)
    for name in displayed_parameters(variable)
        if haskey(parameters, name) && parameters[name].value isa AbstractROI
            push!(rois, RoiParam(name, parameters[name].value))
        end
    end
    PlotSpec(spec.name, spec.layers; spec.title, spec.xlabel, spec.ylabel, spec.fixed_aspect,
             rois, models = vcat(spec.models, models))
end

# Bring the view's spec up to date with its source, rebuilding it if the key of
# what it depends on changed. Returns a message to show instead of the plot when
# there's no spec to draw yet.
function refresh_spec!(view::SpecView, plot::Plot, source::DefaultSpec)
    store = get(state[].client.variable_data, source.variable, nothing)
    if isnothing(store) || store.data isa ArrayMetadata
        "Waiting for data: $(source.variable)"
    elseif ndims(store.data) > 2
        "$(source.variable): unsupported data shape $(typeof(store.data))"
    else
        key = (default_spec_key(store), displayed_parameters(source.variable))
        if key != view.spec_key
            spec = default_spec(source.variable, store)
            set_spec!(view, plot, extend_spec(spec, source.variable, source.models))
            view.spec_key = key
        end
        nothing
    end
end

function refresh_spec!(view::SpecView, plot::Plot, source::CorrelationSpec)
    variable_data = state[].client.variable_data
    x_name, y_name = source.selected
    if isempty(source.variable_names)
        "No scalar or vector variables available to correlate."
    elseif !haskey(variable_data, x_name) || !haskey(variable_data, y_name)
        "Waiting for data: $(x_name), $(y_name)"
    else
        x = variable_data[x_name]
        y = variable_data[y_name]
        if x.type != y.type
            "Both variables must have the same type to correlate against each other."
        else
            key = (x_name, y_name, x.type, x.title, y.title)
            if key != view.spec_key
                set_spec!(view, plot, correlation_spec(x_name, y_name, x, y))
                view.spec_key = key
            end
            nothing
        end
    end
end

# Follows the spec as it changes from train to train, with the @display ROIs of
# the variable advertising it. If the variable stops advertising it the view
# keeps the last one seen.
function refresh_spec!(view::SpecView, plot::Plot, source::AdvertisedSpec)
    store = get(state[].client.variable_data, source.variable, nothing)
    i = isnothing(store) ? nothing : findfirst(spec -> spec.name == source.name, store.plot_specs)
    if !isnothing(i)
        key = (store.plot_specs[i], displayed_parameters(source.variable))
        if key != view.spec_key
            set_spec!(view, plot, extend_spec(store.plot_specs[i], source.variable, ModelOverlay[]))
            view.spec_key = key
        end
    end
    isnothing(view.spec) ? "Waiting for plot spec: $(source.name)" : nothing
end

# Called once per frame before BeginPlot: brings the spec up to date and returns
# what to draw, which is a lone Empty with a message if there's nothing yet or
# the spec doesn't fit the data.
function prepare!(view::SpecView, plot::Plot, updated_variables)
    try
        message = refresh_spec!(view, plot, view.source)
        if isnothing(message)
            frames = PlotType[]
            for layer in view.layers
                layer_frames!(frames, view, layer, updated_variables)
            end
            for model in view.models
                sample_curves!(view, model, updated_variables)
                append!(frames, model.curves)
            end
            frames
        else
            PlotType[Empty(message)]
        end
    catch err
        if err isa SpecError
            PlotType[Empty(err.msg)]
        else
            rethrow()
        end
    end
end

# Upload `data` to the GPU heatmap held by `img` and colormap it, returning the
# Image frame to draw. Reuses cached GPU resources across frames; only re-uploads
# and rescales when the data changed, log mode toggled, or on first use.
function prepare_heatmap!(img::ImageState, data, x_axis, y_axis, was_updated)
    cb = img.colorbar
    ctx = get_heatmap_context()
    needs_initial_upload = isnothing(img.gpu_heatmap)
    if needs_initial_upload
        img.gpu_heatmap = GPUHeatmap()
    end
    gpu = img.gpu_heatmap
    update_colormap!(ctx, img.colormap)

    log = img.log_scale[]
    log_changed = !needs_initial_upload && gpu.log_scale != log
    if was_updated || needs_initial_upload || log_changed || gpu.colormap != img.colormap
        if was_updated || needs_initial_upload
            upload_data!(gpu, data)
        end
        if needs_initial_upload || log_changed || cb.autoscale[]
            dmin, dmax = sampled_pctile!(gpu.hist_buf, data, log)
            cb.clip_min[] = dmin
            cb.clip_max[] = dmax
            # Don't stomp a manual zoom — only reset the visible range
            # if the user has not adjusted it themselves (or just
            # toggled log mode, which makes the old range meaningless).
            if needs_initial_upload || log_changed || !cb.display_zoomed
                margin = 0.1 * (dmax - dmin)
                cb.display_min[] = dmin - margin
                cb.display_max[] = dmax + margin
            end
        end
        render_colormapped!(gpu, ctx, cb.clip_min[], cb.clip_max[], log)
        gpu.log_scale = log
        gpu.colormap = img.colormap
    end
    return Image(data, x_axis, y_axis, gpu)
end

# Derive plot-space axis bounds for an Image frame.
function image_bounds(frame::Image)
    rows, cols = size(frame.data)
    x_min = isnothing(frame.x_axis) ? 0 : first(frame.x_axis)
    x_max = isnothing(frame.x_axis) ? cols : last(frame.x_axis)
    y_min = isnothing(frame.y_axis) ? 0 : first(frame.y_axis)
    y_max = isnothing(frame.y_axis) ? rows : last(frame.y_axis)
    # A degenerate range (all lookup values equal) would give the image zero
    # extent and divide by zero in the hover index math, so fall back to plain
    # pixel indices for that axis.
    if x_max == x_min
        x_min, x_max = 0, cols
    end
    if y_max == y_min
        y_min, y_max = 0, rows
    end
    return (rows, cols, x_min, x_max, y_min, y_max)
end

# Draws a frame with its ImPlot primitive.
function plot_frame!(frame::Line)
    spec = if isnothing(frame.color)
        ImPlot.ImPlotSpec(; FillAlpha=frame.opacity)
    else
        ImPlot.ImPlotSpec(; LineColor=frame.color, FillAlpha=frame.opacity)
    end
    if frame.style === :scatter
        ImPlot.PlotScatter(frame.label, frame.xs, frame.ys; spec)
    else
        ImPlot.PlotLine(frame.label, frame.xs, frame.ys; spec)
    end
end

function plot_frame!(frame::Bars)
    # Black bar outlines
    spec = ImPlot.ImPlotSpec(; LineColor=ig.ImVec4(0, 0, 0, 1))
    ImPlot.PlotBars(frame.label, frame.xs, frame.ys; bar_size=frame.bar_size, spec)
end

function plot_frame!(frame::Band)
    # Same label_id ties the band and line to one legend entry so ImPlot
    # gives them matching colors.
    ImPlot.PlotShaded(frame.label, frame.xs, frame.lower, frame.upper; spec=ImPlot.ImPlotSpec(; FillAlpha=0.5))
    ImPlot.PlotLine(frame.label, frame.xs, frame.line_ys)
end

# The image, with a readout of the pixel under the mouse.
function plot_frame!(frame::Image)
    rows, cols, x_min, x_max, y_min, y_max = image_bounds(frame)
    tex_ref = ig.ImTextureRef(ig.ImTextureID(frame.gpu.output_tex))

    # ImGui 1.92's GL backend binds a linear sampler for every draw, which
    # overrides our texture's GL_NEAREST filter and blurs the heatmap when
    # scaled. Switch the plot draw list to nearest sampling around the image,
    # then restore linear for everything drawn afterwards.
    draw_list = ImPlot.GetPlotDrawList()
    platform_io = ig.GetPlatformIO()
    set_nearest = unsafe_load(platform_io.DrawCallback_SetSamplerNearest)
    set_linear = unsafe_load(platform_io.DrawCallback_SetSamplerLinear)
    if set_nearest != C_NULL
        ig.AddCallback(draw_list, set_nearest)
    end

    # Matplotlib convention: first dim = row (vertical, top→bottom),
    # second dim = col (horizontal, left→right). data[1,1] at plot top-left;
    # data[rows,cols] at plot bottom-right. Y axis is inverted so y_min sits
    # at the top — pass swapped y bounds.
    ImPlot.PlotImage("", tex_ref,
                     ImPlot.ImPlotPoint(x_min, y_max),
                     ImPlot.ImPlotPoint(x_max, y_min))

    if set_linear != C_NULL
        ig.AddCallback(draw_list, set_linear)
    end

    if ImPlot.IsPlotHovered()
        mouse = ImPlot.GetPlotMousePos()
        j = floor(Int, (mouse.x - x_min) / (x_max - x_min) * cols) + 1
        i = floor(Int, (mouse.y - y_min) / (y_max - y_min) * rows) + 1
        if 1 <= i <= rows && 1 <= j <= cols
            val = frame.data[i, j]
            ImPlot.AnnotationClamped(mouse.x, mouse.y, ImVec2(10, -10), "[$i, $j] $val")
        end
    end
end

function side_panel(view::SpecView)
    id = view.id
    client = state[].client
    # The compression of the first layer's variable, unless it's paired with another
    if !isempty(view.layers) && isnothing(view.layers[1].spec.lookup)
        name = view.layers[1].spec.data
        store = get(client.variable_data, name, nothing)
        if !isnothing(store) && !(store.data isa CircularBuffer) && ig.CollapsingHeader("Compression##$(id)")
            draw_compression_settings(id, name, Ref(Cfloat(client.subscriptions[name].k)), store)
        end
    end
    # The fit follows the first layer, and there's nothing to fit on an image
    if !isempty(view.layers) && view.layers[1].spec.mark == Mark_Rect
        ig.BeginDisabled()
        ig.CollapsingHeader("Fitting##$(id)")
        ig.EndDisabled()
    else
        draw_fitting_settings(id, view.fit)
    end
end

function bottom_controls(view::SpecView)
    id = view.id
    variable_data = state[].client.variable_data

    # Scalar histories accumulate, along with what was paired from them
    histories = [variable_data[name] for name in view.subscribed
                 if haskey(variable_data, name) && variable_data[name].data isa CircularBuffer]
    if !isempty(histories)
        ig.SameLine()
        if ig.Button("Clear##$(id)")
            foreach(clear_variable_data, histories)
            clear_paired_data!(view)
        end
    end

    if any(layer -> !isnothing(layer.matcher) && layer.spec.lookup.key == LookupKey_TrainId, view.layers)
        ig.SameLine()
        ig.SetNextItemWidth(135)
        if ig.DragFloat("Binning resolution##$(id)",
                        view.binning_resolution, 0.01f0,
                        0.0f0, typemax(Cfloat), "%.12f",
                        ig.ImGuiSliderFlags_AlwaysClamp)
            view.resolution_touched = true
        end
    end

    image = image_state(view)
    if !isnothing(image)
        image_controls(image, id)
    end
end

# Bottom-row image controls (fixed aspect / auto colorbar / log colormap). Each
# is prefixed with SameLine so it sits alongside the autoscale/log buttons.
function image_controls(img::ImageState, id::String)
    ig.SameLine()
    ig.Checkbox("Fixed aspect##$(id)", img.fixed_aspect)
    ig.SameLine()
    if ig.Checkbox("Auto colorbar##$(id)", img.colorbar.autoscale)
        if img.colorbar.autoscale[]
            img.colorbar.display_zoomed = false
        end
    end
    ig.SameLine()
    ig.Checkbox("Log colormap##$(id)", img.log_scale)
end

# Begin the plot-area child, shrunk to leave room for the side panel when open.
# Returns (plot_size, plot_area_h) for use inside the child. Must be paired
# with end_plot_area!.
function begin_plot_area!(plot, side_panel_width, bottom_row_h=30)
    region_avail = ig.GetContentRegionAvail()
    plot_area_h = max(region_avail.y - bottom_row_h, 100)
    spacing = unsafe_load(ig.GetStyle().ItemSpacing.x)
    plot_w = plot.show_side_panel[] ?
        max(region_avail.x - side_panel_width - spacing, 100f0) : region_avail.x

    ig.BeginChild("##plot-area-$(plot.id)", ImVec2(plot_w, plot_area_h))
    inner_avail = ig.GetContentRegionAvail()
    return ImVec2(inner_avail.x, inner_avail.y), plot_area_h
end

# Closes the plot-area child opened by begin_plot_area!, then if the panel is
# open draws it alongside via the caller-supplied `draw_panel`.
function end_plot_area!(plot, side_panel_width, plot_area_h, draw_panel)
    ig.EndChild()

    if plot.show_side_panel[]
        ig.SameLine()
        if ig.BeginChild("##sidepanel-$(plot.id)", ImVec2(side_panel_width, plot_area_h),
                         ig.ImGuiChildFlags_Borders)
            draw_panel()
        end
        ig.EndChild()
    end
end

# The state of the first image the view draws, if any. It's the one the colorbar
# and the image controls act on.
function image_state(view::SpecView)
    i = findfirst(layer -> !isnothing(layer.image) && !isnothing(layer.image.gpu_heatmap), view.layers)
    isnothing(i) ? nothing : view.layers[i].image
end

function draw_plot(plot::Plot, updated_variables)
    ig.SetNextWindowSize((800, 500), ig.ImGuiCond_FirstUseEver)
    side_panel_width = 320f0
    colorbar_width = 100f0

    view = plot.view
    title = isnothing(view.spec) ? "" : view.spec.title
    win_id = isempty(title) ? plot.id : "$(title)##$(plot.id)"

    if ig.Begin(win_id, plot.open)
        plot.dock_id = ig.GetWindowDockID()

        top_controls(view.source, view.id)

        plot_size, plot_area_h = begin_plot_area!(plot, side_panel_width)

        frames = prepare!(view, plot, updated_variables)
        all_empty = all(f -> f isa Empty, frames)

        if all_empty
            for f in frames
                if !isempty(f.message)
                    ig.TextWrapped(f.message)
                end
            end
        else
            apply_autoscale(plot)

            # Reserve room on the right for the side-panel tab button.
            tab_w = 14.0f0
            spacing = unsafe_load(ig.GetStyle().ItemSpacing.x)
            img = image_state(view)
            reserved = tab_w + spacing + (isnothing(img) ? 0f0 : colorbar_width + spacing)
            plot_width = max(plot_size.x - reserved, 100f0)
            plot_flags = (!isnothing(img) && img.fixed_aspect[]) ?
                         ImPlot.ImPlotFlags_Equal : ImPlot.ImPlotFlags_None

            if ImPlot.BeginPlot(plot.id, ImVec2(plot_width, plot_size.y), plot_flags)
                ImPlot.SetupAxis(ImPlot.ImAxis_X1, axis_label(view.spec, view.spec.xlabel, :x))
                ImPlot.SetupAxis(ImPlot.ImAxis_Y1, axis_label(view.spec, view.spec.ylabel, :y))
                ImPlot.SetupAxisFormat(ImPlot.ImAxis_X1, "%.9g")
                ImPlot.SetupAxisFormat(ImPlot.ImAxis_Y1, "%.9g")
                apply_log_scales(plot)
                for f in frames
                    if !(f isa Empty)
                        plot_frame!(f)
                    end
                end
                draw_view_overlays(view, frames)
                check_plot_interaction!(plot)
                ImPlot.EndPlot()
            end

            if !isnothing(img)
                ig.SameLine()
                if interactive_colorbar(img, view.id, ImVec2(colorbar_width, plot_size.y))
                    cb = img.colorbar
                    update_colormap!(get_heatmap_context(), img.colormap)
                    render_colormapped!(img.gpu_heatmap, get_heatmap_context(),
                                        cb.clip_min[], cb.clip_max[], img.log_scale[])
                end
            end

            side_panel_tab(plot, tab_w, plot_size.y)
        end

        end_plot_area!(plot, side_panel_width, plot_area_h, () -> side_panel(view))

        if !all_empty
            autoscale_buttons(plot)
            ig.SameLine()
            log_scale_buttons(plot)
            bottom_controls(view)
        end
    end

    ig.End()
end

function var_type_label(store)
    if store.type == VariableType_Scalar
        "scalar"
    elseif store.type == VariableType_Vector
        "vector"
    elseif store.type == VariableType_Array
        sz = store.data isa ArrayMetadata ? store.data.size : size(store.data)
        """array $(join(sz, "×"))"""
    else
        ""
    end
end

# A combo picking one of `var_names`. Returns the selected name.
function var_combo(label, selected, var_names, variable_data)
    preview = if haskey(variable_data, selected)
        "$(selected)  ($(var_type_label(variable_data[selected])))"
    else
        selected
    end
    ig.SetNextItemWidth(250)

    if ig.BeginCombo(label, preview)
        for name in var_names
            is_selected = name == selected
            if ig.Selectable(name, is_selected)
                selected = name
            end

            ig.SameLine()

            ig.TextDisabled(var_type_label(variable_data[name]))
            if is_selected
                ig.SetItemDefaultFocus()
            end
        end

        ig.EndCombo()
    end

    return selected
end

# Widgets at the top of the plot window, above the plot area.
top_controls(::SpecSource, id) = nothing

# The X/Y variable pickers. The view rebuilds its spec from the selection, see
# refresh_spec!.
function top_controls(source::CorrelationSpec, id)
    variable_data = state[].client.variable_data

    empty!(source.variable_names)
    for (name, variable) in variable_data
        if variable.type in (VariableType_Scalar, VariableType_Vector)
            push!(source.variable_names, name)
        end
    end
    sort!(source.variable_names)

    # Seed the selection on first use. A selected variable that goes away (e.g.
    # over a context reload) stays selected, so it's restored once it reappears.
    if !isempty(source.variable_names)
        for i in eachindex(source.selected)
            if isempty(source.selected[i])
                source.selected[i] = source.variable_names[1]
            end
        end
    end

    # The paired history follows the swap, see carry_matcher
    if ig.Button("Swap axes##$(id)")
        reverse!(source.selected)
    end

    ig.SameLine()
    source.selected[1] = var_combo("X##corr-x-$(id)", source.selected[1], source.variable_names, variable_data)
    ig.SameLine()
    source.selected[2] = var_combo("Y##corr-y-$(id)", source.selected[2], source.variable_names, variable_data)
end
