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
    colormap_id::Int  # which ImPlot colormap is currently uploaded (-1 = none)

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
        colormap_tex, -1,
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

# ImPlot colormap index of the custom Turbo colormap, set by add_turbo_colormap()
TURBO_COLORMAP::Cint = -1

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

# Register the Turbo colormap with ImPlot. Must be called after
# ImPlot.CreateContext().
function add_turbo_colormap()
    global TURBO_COLORMAP = ImPlot.AddColormap("Turbo", TURBO_COLORMAP_DATA,
                                               length(TURBO_COLORMAP_DATA), false)
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

    return GPUHeatmap(data_tex, output_tex, fbo, 0, 0, false, UInt8[], Int32[], false)
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
Convert matrix data into the reusable `convert_buf`, reinterpreted as a matrix
of the target GL type. Returns either the original data (if no conversion
needed) or a zero-copy view over the resized buffer.
"""
function prepare_data!(h::GPUHeatmap, data::AbstractMatrix{T}) where T
    if T <: GLNativeTypes
        return data
    end

    # Convert into the cached byte buffer to avoid per-frame allocations
    G = gl_convert_type(T)
    nbytes = length(data) * sizeof(G)
    resize!(h.convert_buf, nbytes)
    buf = unsafe_wrap(Matrix{G}, Ptr{G}(pointer(h.convert_buf)), size(data))
    copyto!(buf, data)

    return buf
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
# mode, which have no real log). Percentiles are invariant under a monotonic
# transform, so log mode just log10's the two final results rather than
# transforming every sample. `buf` is the reused bin-count scratch.
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
    if !(hi > lo)
        if log
            return hi > 0 ? (log10(hi), log10(hi)) : (0.0, 1.0)
        end
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
        x = data[i]
        if log ? (isfinite(x) && x > 0) : isfinite(x)
            b = clamp(floor(Int, (x - lo) * scale) + 1, 1, PCTILE_NBINS)
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
    return log ? (log10(p1), log10(p99)) : (p1, p99)
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

# Per-layer fit configuration. Used by both VariableLayer and CorrelationLayer
# so the side-panel fitting UI can be driven from a single struct.
@kwdef mutable struct FitSettings
    fit_type::Ref{Cint} = Ref(Cint(0))
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

# Re-run the selected fit against the plot's current X/Y samples. Called from
# the draw_plot data-update path so the popt stays in sync with what's shown.
function compute_fit!(fit::FitSettings, ydata::AbstractVector,
                      xdata::Maybe{AbstractVector}=nothing;
                      sigma::Maybe{AbstractVector}=nothing)
    name = FIT_TYPES[fit.fit_type[] + 1]
    fixed = fixed_vector(fit)
    t0 = time_ns()
    if name == "Line"
        fit.popt, fit.retcode = fit_line(ydata, xdata; sigma, fixed)
    elseif name == "Gaussian"
        fit.popt, fit.retcode = fit_gaussian(ydata, xdata; sigma, fixed)
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

# Overlay the fitted model curve on the current ImPlot plot, if any.
function draw_fit_overlay(fit::FitSettings)
    if !isempty(fit.model_x)
        name = FIT_TYPES[fit.fit_type[] + 1]
        ImPlot.PlotLine("$(name) fit", fit.model_x, fit.model_y)
    end
end

# --- Plot type payloads ---
#
# A `PlotType` is what a Layer hands back from `prepare!` each frame — a
# description of what to draw. The Plot widget dispatches on the concrete type
# to pick the right ImPlot primitive.

abstract type PlotType end

# 1D series. `style` selects the ImPlot primitive:
#   :line    → PlotLine
#   :scatter → PlotScatter
struct Line <: PlotType
    xs
    ys
    label::String
    style::Symbol
    # Explicit per-series color, used by SpecLayer fan-out (e.g. a gradient over
    # grouped series). nothing lets ImPlot cycle its palette as usual.
    color::Maybe{ig.ImVec4}
end
Line(xs, ys, label, style) = Line(xs, ys, label, style, nothing)

# Bar series. Separate from Line because it uses PlotBars/PlotBarsH and carries
# a bar_size. Covers histograms and any vector drawn as bars.
struct Bars <: PlotType
    xs
    ys
    label::String
    bar_size::Float64
    horizontal::Bool
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

# Colormapped 2D data, rendered via the GPU heatmap path. `x_axis`/`y_axis`
# may be nothing (defaults to pixel coords).
struct Image <: PlotType
    data
    x_axis::Maybe{AbstractVector}
    y_axis::Maybe{AbstractVector}
end

# Layer has nothing to draw this frame. `message` (when non-empty) is surfaced
# in place of the plot — used for "types must match" and similar diagnostics.
struct Empty <: PlotType
    message::String
end

# --- Layer interface ---
#
# A Layer is a self-contained piece of a Plot. Plot owns Vector{Layer} and
# orchestrates BeginPlot/EndPlot; each layer contributes one PlotType and
# optional overlays/side-panel/bottom-row widgets.

abstract type Layer end

# Called once per frame before BeginPlot. May mutate caches, run fits, push
# subscriptions, etc. Returns what to draw this frame.
function prepare! end

# Called inside BeginPlot/EndPlot, after the primitive has been drawn.
# Used for overlays that need plot-space coords: fit curves, ROI rects, hover
# annotations.
draw_overlay(::Layer, plot, ::PlotType) = nothing

# Widgets rendered at the top of the plot window, above the plot area. Used by
# layers that own variable-selection UI (e.g. correlation X/Y combos).
top_controls(::Layer, plot) = nothing

# Collapsing header(s) for this layer in the side panel.
side_panel(::Layer, plot) = nothing

# Extra widgets in the bottom row, right of the shared autoscale/log buttons.
bottom_controls(::Layer, plot) = nothing

# (xlabel, ylabel) this layer wants. Either may be "" to abstain — first
# non-empty wins at the plot level.
axis_labels(::Layer) = ("", "")

# Title contribution for the window — same first-wins rule.
window_title(::Layer) = ""

# Reset any accumulated state (e.g. paired history) when the user hits Clear.
# Default: no-op.
clear_layer_data!(::Layer) = nothing

# Unique-within-plot string used as the `##` suffix on this layer's ImGui
# widgets. Assigned when the layer is added to a Plot.
function layer_id end

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
    const colorbar::ColorbarState = ColorbarState()
    gpu_heatmap::Union{Nothing, GPUHeatmap} = nothing
    # ROI parameter values updated locally during a drag, keyed by parameter
    # name. Flushed to the engine when the user releases the mouse so we don't
    # flood it with per-frame updates.
    const pending_roi_updates::Dict{String, RectROI} = Dict{String, RectROI}()
end

# Pairs samples from two VariableData stores on matching train IDs. Owns the
# paired history buffers and an optional binning accumulator. Pure data
# plumbing — no ImGui state.
@kwdef mutable struct VariableTrainmatcher
    const x_data::Vector{Float64} = Float64[]
    const y_data::Vector{Float64} = Float64[]
    accu::Maybe{AccuPairSequence} = nothing
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
    new_tids = get(updated_variables, x_name, Set{Int}())
    if haskey(updated_variables, y_name)
        union!(new_tids, updated_variables[y_name])
    end

    appended = false
    for tid in new_tids
        xi = findfirst(==(tid), x_store.scalar_tids)
        yi = findfirst(==(tid), y_store.scalar_tids)
        if !isnothing(xi) && !isnothing(yi)
            xv = x_store.data[xi]
            yv = y_store.data[yi]
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
    if res > 0 && (isnothing(m.accu) || m.accu.resolution != res)
        m.accu = AccuPairSequence(m.x_data, m.y_data, res)
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

# A layer that subscribes to one variable and renders its data as a line, bar
# series, scalar buffer, or matrix. `image` is allocated lazily the first time
# the variable's data turns out to be a matrix.
@kwdef mutable struct VariableLayer <: Layer
    const name::String
    const layer_id_str::String
    const precision::Ref{Cint} = Ref(Cint(-1))
    const fit::FitSettings = FitSettings()
    image::Maybe{ImageState} = nothing
end

layer_id(layer::VariableLayer) = layer.layer_id_str

# A layer that pairs samples from two variables on matching train IDs and
# renders the result as a scatter (or shaded band when binning is enabled).
@kwdef mutable struct CorrelationLayer <: Layer
    const layer_id_str::String
    const x_var::Ref{Cint} = Ref(Cint(0))
    const y_var::Ref{Cint} = Ref(Cint(0))
    const subscribed::Vector{String} = ["", ""]
    const binning_resolution::Ref{Cfloat} = Ref(Cfloat(0))
    const matcher::VariableTrainmatcher = VariableTrainmatcher()
    const fit::FitSettings = FitSettings()
    # Refreshed each frame from client.variable_data; used by the X/Y combos.
    const variable_names::Vector{String} = String[]
end

layer_id(layer::CorrelationLayer) = layer.layer_id_str

# Renders an engine-advertised PlotSpec. Each frame it looks up the latest spec
# (by name) from the source variable's store, reconciles its subscriptions to
# the variables the spec references, and emits one or more PlotTypes. A
# LayerSpec whose grouping channel (color) is bound to a dimension fans out into
# one series per coordinate along that dim. Self-contained: it owns its
# subscriptions and tracks the spec across trains. If the variable stops
# advertising the spec, the last-seen layout is kept (the plot freezes rather
# than vanishing). Frame drawing reuses the generic plot_frame! methods.
@kwdef mutable struct SpecLayer <: Layer
    const layer_id_str::String
    const source_var::String           # variable advertising the spec
    const spec_name::String            # which PlotSpec, by name
    const subscribed::Set{String} = Set{String}()
    image::Maybe{ImageState} = nothing
    last_spec::Maybe{PlotSpec} = nothing
end

layer_id(layer::SpecLayer) = layer.layer_id_str

@kwdef mutable struct Plot
    const id::String
    const open::Ref{Bool} = Ref(true)
    const autoscale_x::Ref{Bool} = Ref(true)
    const autoscale_y::Ref{Bool} = Ref(true)
    const log_x::Ref{Bool} = Ref(false)
    const log_y::Ref{Bool} = Ref(false)
    const show_side_panel::Ref{Bool} = Ref(false)
    const layers::Vector{Layer} = Layer[]
    dock_id::UInt32 = 0
end

# Build a VariableLayer wired to `name`, suitable for pushing onto `plot.layers`.
# Seeds precision from any existing subscription and bumps the subscription count.
function VariableLayer(plot::Plot, name::AbstractString)
    subscriptions = state[].client.subscriptions
    precision = haskey(subscriptions, name) ? subscriptions[name].precision : -1
    layer = VariableLayer(;
        name = String(name),
        layer_id_str = "$(plot.id)/$(name)/$(length(plot.layers) + 1)",
        precision = Ref(Cint(precision)),
    )
    subscribe_variable(state[], name; precision)
    return layer
end

# Convenience: a Plot containing a single VariableLayer for `name`.
variable_plot(name::AbstractString, counter::Int) =
    variable_plot(name, "$(name)##plot-$(counter)")

function variable_plot(name::AbstractString, id::String, dock_id = 0)
    plot = Plot(; id, dock_id = UInt32(dock_id))
    push!(plot.layers, VariableLayer(plot, name))
    return plot
end

# A Plot containing a single CorrelationLayer.
function correlation_plot(counter::Integer)
    id = "CorrelationPlot##plot-$(counter)"
    plot = Plot(; id)
    push!(plot.layers, CorrelationLayer(; layer_id_str = "$(id)/correlation/1"))
    return plot
end

function correlation_plot(id::String, dock_id::Integer = 0)
    plot = Plot(; id, dock_id = UInt32(dock_id))
    push!(plot.layers, CorrelationLayer(; layer_id_str = "$(id)/correlation/1"))
    return plot
end

# A Plot rendering a single advertised PlotSpec (by name) from `source_var`.
function spec_plot(source_var::AbstractString, spec_name::AbstractString, counter::Integer)
    id = "$(source_var)/$(spec_name)##plot-$(counter)"
    plot = Plot(; id)
    push!(plot.layers, SpecLayer(; layer_id_str = "$(id)/spec",
                                 source_var = String(source_var), spec_name = String(spec_name)))
    return plot
end

function Base.close(plot::Plot)
    for layer in plot.layers
        close(layer)
    end
end

function Base.close(layer::VariableLayer)
    if !isnothing(layer.image) && !isnothing(layer.image.gpu_heatmap)
        destroy!(layer.image.gpu_heatmap)
        layer.image.gpu_heatmap = nothing
    end
    unsubscribe_variable(state[], layer.name)
end

function Base.close(layer::CorrelationLayer)
    unsubscribe_variable(state[], layer.subscribed[1])
    unsubscribe_variable(state[], layer.subscribed[2])
end

clear_plot(plot::Plot) = foreach(clear_layer_data!, plot.layers)
clear_layer_data!(layer::CorrelationLayer) = empty!(layer.matcher)

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

# zfp precision input, ratio, and throughput readout. `id` namespaces the
# imgui widgets, `name` is the qualified variable name to retune.
function draw_compression_settings(id, name, precision::Ref{Cint}, store)
    compressed = isfinite(store.compression_ratio)
    enabled = compressed && store.compress
    if !enabled
        ig.BeginDisabled()
    end
    ig.SetNextItemWidth(120)
    if ig.InputInt("zfp precision##$(id)", precision)
        set_subscription_precision(state[], name, Int(precision[]))
    end
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
function interactive_colorbar(layer::VariableLayer, size::ImVec2)
    img = layer.image
    cb = img.colorbar
    display_min = cb.display_min[]
    display_max = cb.display_max[]
    clip_min = cb.clip_min[]
    clip_max = cb.clip_max[]
    # In log mode all four refs hold log10(value); the colorbar renders that
    # space directly and tick labels read as exponents.
    tick_format = img.log_scale[] ? "1e%g" : "%g"
    id = layer.layer_id_str

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
                         TURBO_COLORMAP)
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

# Treat an all-zero RectROI as uninitialized and seed a centered default
# covering 1/4 of the visible area.
function default_roi(x_min, x_max, y_min, y_max, idx=1)
    w = (x_max - x_min) / 2
    h = (y_max - y_min) / 2
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    # Stagger successive ROIs diagonally so they don't fully overlap.
    step = (idx - 1) * 0.05
    dx = (x_max - x_min) * step
    dy = (y_max - y_min) * step
    RectROI(cx - w / 2 + dx, cy - h / 2 + dy, w, h)
end

# Draw any RectROI parameters associated with `layer.name` via @display as
# draggable rectangles on top of the image, with the parameter name labelled
# above the top-left corner. Sends a ChangeParameter when the user drags.
function draw_roi_overlays(layer::VariableLayer, x_min, x_max, y_min, y_max)
    client = state[].client
    displays = get(client.context.displays, layer.name, String[])
    pending = layer.image.pending_roi_updates
    for (idx, param_name) in enumerate(displays)
        param = get(client.context.parameters, param_name, nothing)
        if isnothing(param) || !(param.value isa RectROI)
            continue
        end
        roi = param.value
        if !isassigned(roi)
            roi = default_roi(x_min, x_max, y_min, y_max, idx)
        end
        x1 = Ref(Cdouble(roi.corner_x))
        y1 = Ref(Cdouble(roi.corner_y))
        x2 = Ref(Cdouble(roi.corner_x + roi.width))
        y2 = Ref(Cdouble(roi.corner_y + roi.height))
        col = ROI_COLORS[mod1(idx, length(ROI_COLORS))]
        rect_col = ImVec4(col.x, col.y, col.z, 0.75)
        id = int32_hash(layer.layer_id_str, param_name)
        held = Ref(false)

        if ImPlot.DragRect(id, x1, y1, x2, y2, rect_col, 0, C_NULL, C_NULL, held)
            xlo, xhi = minmax(x1[], x2[])
            ylo, yhi = minmax(y1[], y2[])
            new_roi = RectROI(xlo, ylo, xhi - xlo, yhi - ylo)
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

        # Label above the top-left corner. Image plots invert the Y axis so
        # y_min is the top, meaning the ROI top edge is at min(y1, y2).
        # PlotText centers on (x, y), so shift right by half the text width to
        # left-align at xlo, and up by half a line so the text sits above the
        # rect.
        xlo = min(x1[], x2[])
        ytop = min(y1[], y2[])
        base_size = unsafe_load(ig.GetStyle()).FontSizeBase
        ig.PushFont(C_NULL, base_size * 2)
        text_size = ig.CalcTextSize(param_name)
        ImPlot.PushStyleColor(ImPlot.ImPlotCol_InlayText, col)
        ImPlot.PlotText(param_name, Cdouble(xlo), Cdouble(ytop),
                        ImVec2(text_size.x / 2, -text_size.y / 2 - 2))
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
            ig.SetNextItemWidth(120)
            # InputDouble's bound buffer updates per keystroke (only the commit
            # to `param.value` is gated on Enter / focus-loss), so comparing
            # edit_buf to value detects uncommitted changes mid-edit.
            uncommitted = param.fixed && param.edit_buf[] != param.value
            if uncommitted
                ig.PushStyleColor(ig.ImGuiCol_FrameBg, ig.IM_COL32(143, 98, 0, 255))
            end
            ig.InputDouble("$(pname)", param.edit_buf, 0.0, 0.0, "%g", flags)
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

# --- VariableLayer methods ---

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

function prepare!(layer::VariableLayer, plot::Plot, updated_variables)
    client = state[].client
    store = get(client.variable_data, layer.name, nothing)
    if isnothing(store) || store.data isa ArrayMetadata
        return Empty("Waiting for data: $(layer.name)")
    end
    data = store.data
    if !(eltype(data) <: Real)
        return Empty("$(layer.name): unsupported array type $(typeof(data))")
    elseif isempty(data)
        return Empty("$(layer.name): array has length 0, nothing to plot")
    end

    was_updated = haskey(updated_variables, layer.name)

    if data isa AbstractVector
        xs, ys = if data isa CircularBuffer
            store.scalar_tids_cache, store.scalar_data_cache
        elseif !isnothing(store.x_axis)
            store.x_axis, data
        elseif data isa DimArray
            parent(lookup(data)[1]), parent(data)
        else
            1:length(data), data
        end
        if was_updated
            compute_fit!(layer.fit, ys, xs)
        end
        if store.plot_type === :histogram
            bar_size = length(xs) > 1 ? Float64(abs(xs[2] - xs[1])) : 1.0
            return Bars(xs, ys, store.title, bar_size, false)
        else
            style = length(ys) == 1 ? :scatter : :line
            return Line(xs, ys, store.title, style)
        end
    elseif data isa AbstractMatrix
        if isnothing(layer.image)
            layer.image = ImageState()
            layer.image.fixed_aspect[] = store.fixed_aspect
        end
        # Fall back to a 2D DimArray's dim lookups for the axes (dim 1 → Y/rows,
        # dim 2 → X/cols), mirroring the vector branch above.
        xax = if !isnothing(store.x_axis)
            store.x_axis
        elseif data isa DimArray
            forward_lookup(data, 2)
        else
            nothing
        end
        yax = if !isnothing(store.y_axis)
            store.y_axis
        elseif data isa DimArray
            forward_lookup(data, 1)
        else
            nothing
        end
        return prepare_heatmap!(layer.image, data, xax, yax, was_updated)
    end
    return Empty("$(layer.name): unsupported data shape $(typeof(data))")
end

# Upload `data` to the GPU heatmap held by `img` and colormap it, returning the
# Image frame to draw. Reuses cached GPU resources across frames; only re-uploads
# and rescales when the data changed, log mode toggled, or on first use. Shared
# by VariableLayer and SpecLayer.
function prepare_heatmap!(img::ImageState, data, x_axis, y_axis, was_updated)
    cb = img.colorbar
    ctx = get_heatmap_context()
    needs_initial_upload = isnothing(img.gpu_heatmap)
    if needs_initial_upload
        img.gpu_heatmap = GPUHeatmap()
    end
    gpu = img.gpu_heatmap
    update_colormap!(ctx, TURBO_COLORMAP)

    log = img.log_scale[]
    log_changed = !needs_initial_upload && gpu.log_scale != log
    if was_updated || needs_initial_upload || log_changed
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
    end
    return Image(data, x_axis, y_axis)
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

# Generic line/bar drawing, shared by every layer (CorrelationLayer overrides
# Line with its own alpha-blended scatter). `frame.color`, when set, fixes the
# series colour — used by SpecLayer fan-out; otherwise ImPlot cycles its palette.
function plot_frame!(::Layer, frame::Line)
    spec = isnothing(frame.color) ? ImPlot.ImPlotSpec() : ImPlot.ImPlotSpec(; LineColor=frame.color)
    if frame.style === :scatter
        ImPlot.PlotScatter(frame.label, frame.xs, frame.ys; spec)
    else
        ImPlot.PlotLine(frame.label, frame.xs, frame.ys; spec)
    end
end

function plot_frame!(::Layer, frame::Bars)
    # Black bar outlines; horizontal orientation via the bars flag.
    flags = frame.horizontal ? ImPlot.ImPlotBarsFlags_Horizontal : ImPlot.ImPlotBarsFlags_None
    spec = ImPlot.ImPlotSpec(; LineColor=ig.ImVec4(0, 0, 0, 1), Flags=Cint(flags))
    ImPlot.PlotBars(frame.label, frame.xs, frame.ys; bar_size=frame.bar_size, spec)
end

function draw_image_frame(gpu, frame::Image)
    _, _, x_min, x_max, y_min, y_max = image_bounds(frame)
    tex_ref = ig.ImTextureRef(ig.ImTextureID(gpu.output_tex))

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
end

plot_frame!(layer::Union{VariableLayer, SpecLayer}, frame::Image) = draw_image_frame(layer.image.gpu_heatmap, frame)

function draw_overlay(layer::VariableLayer, ::Plot, ::Line)
    draw_fit_overlay(layer.fit)
    draw_variable_overlays(layer.name)
end

function draw_overlay(layer::VariableLayer, ::Plot, ::Bars)
    draw_fit_overlay(layer.fit)
    draw_variable_overlays(layer.name)
end

function draw_overlay(layer::VariableLayer, ::Plot, frame::Image)
    rows, cols, x_min, x_max, y_min, y_max = image_bounds(frame)
    draw_roi_overlays(layer, x_min, x_max, y_min, y_max)

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

function side_panel(layer::VariableLayer, ::Plot)
    client = state[].client
    store = get(client.variable_data, layer.name, nothing)
    isnothing(store) && return
    is_scalar = store.data isa CircularBuffer
    is_matrix = store.data isa AbstractMatrix
    id = layer.layer_id_str
    if !is_scalar && ig.CollapsingHeader("Compression##$(id)")
        draw_compression_settings(id, layer.name, layer.precision, store)
    end
    if is_matrix
        ig.BeginDisabled()
        ig.CollapsingHeader("Fitting##$(id)")
        ig.EndDisabled()
    else
        draw_fitting_settings(id, layer.fit)
    end
end

function bottom_controls(layer::VariableLayer, ::Plot)
    client = state[].client
    store = get(client.variable_data, layer.name, nothing)
    isnothing(store) && return
    data = store.data
    id = layer.layer_id_str
    if data isa CircularBuffer
        ig.SameLine()
        if ig.Button("Clear##$(id)")
            clear_variable_data(store)
        end
    end
    if data isa AbstractMatrix && !isnothing(layer.image)
        image_controls(layer.image, id)
    end
end

# Bottom-row image controls (fixed aspect / auto colorbar / log colormap),
# shared by any layer that owns an ImageState. Each is prefixed with SameLine
# so it sits alongside the shared autoscale/log buttons.
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

function axis_labels(layer::VariableLayer)
    client = state[].client
    store = get(client.variable_data, layer.name, nothing)
    isnothing(store) ? ("", "") : (store.xlabel, store.ylabel)
end

function window_title(layer::VariableLayer)
    client = state[].client
    store = get(client.variable_data, layer.name, nothing)
    isnothing(store) ? "" : store.title
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

# Locate the (at most one) image layer in this plot. Returns nothing if no
# layer has allocated an ImageState.
function image_layer(plot::Plot)
    for L in plot.layers
        if (L isa VariableLayer || L isa SpecLayer) && !isnothing(L.image)
            return L
        end
    end
    return nothing
end

# A layer's prepare! may return a single PlotType or several (SpecLayer fans a
# grouped LayerSpec out into one series per slice). Normalise to a vector so the
# draw loop can treat every layer uniformly.
as_frames(f::PlotType) = PlotType[f]
as_frames(fs::AbstractVector{<:PlotType}) = fs

function draw_plot(plot::Plot, updated_variables)
    ig.SetNextWindowSize((800, 500), ig.ImGuiCond_FirstUseEver)
    side_panel_width = 300f0
    colorbar_width = 100f0

    # Build window title from the first layer that has an opinion.
    title = ""
    for L in plot.layers
        t = window_title(L)
        if !isempty(t)
            title = t
            break
        end
    end
    win_id = isempty(title) ? plot.id : "$(title)##$(plot.id)"

    if ig.Begin(win_id, plot.open)
        plot.dock_id = ig.GetWindowDockID()

        for L in plot.layers
            top_controls(L, plot)
        end

        plot_size, plot_area_h = begin_plot_area!(plot, side_panel_width)

        # One frame list per layer, parallel to plot.layers.
        layer_frames = [as_frames(prepare!(L, plot, updated_variables)) for L in plot.layers]
        all_empty = all(f isa Empty for fs in layer_frames for f in fs)

        if all_empty
            for fs in layer_frames
                for f in fs
                    if f isa Empty && !isempty(f.message)
                        ig.TextWrapped(f.message)
                        shown_any = true
                    end
                end
            end
        else
            apply_autoscale(plot)

            # Reserve room on the right for the side-panel tab button.
            tab_w = 14.0f0
            spacing = unsafe_load(ig.GetStyle().ItemSpacing.x)
            img_layer = image_layer(plot)
            reserved = tab_w + spacing + (isnothing(img_layer) ? 0f0 : colorbar_width + spacing)
            plot_width = max(plot_size.x - reserved, 100f0)
            plot_flags = (!isnothing(img_layer) && img_layer.image.fixed_aspect[]) ?
                         ImPlot.ImPlotFlags_Equal : ImPlot.ImPlotFlags_None

            # First non-empty axis label wins.
            xlabel = ylabel = ""
            for L in plot.layers
                xl, yl = axis_labels(L)
                if isempty(xlabel)
                    xlabel = xl
                end
                if isempty(ylabel)
                    ylabel = yl
                end
            end

            if ImPlot.BeginPlot(plot.id, ImVec2(plot_width, plot_size.y), plot_flags)
                ImPlot.SetupAxis(ImPlot.ImAxis_X1, xlabel)
                ImPlot.SetupAxis(ImPlot.ImAxis_Y1, ylabel)
                apply_log_scales(plot)
                for (L, fs) in zip(plot.layers, layer_frames)
                    for f in fs
                        if !(f isa Empty)
                            plot_frame!(L, f)
                            draw_overlay(L, plot, f)
                        end
                    end
                end
                check_plot_interaction!(plot)
                ImPlot.EndPlot()
            end

            if !isnothing(img_layer)
                ig.SameLine()
                if interactive_colorbar(img_layer, ImVec2(colorbar_width, plot_size.y))
                    img = img_layer.image
                    cb = img.colorbar
                    render_colormapped!(img.gpu_heatmap, get_heatmap_context(),
                                        cb.clip_min[], cb.clip_max[], img.log_scale[])
                end
            end

            side_panel_tab(plot, tab_w, plot_size.y)
        end

        end_plot_area!(plot, side_panel_width, plot_area_h,
                       () -> foreach(L -> side_panel(L, plot), plot.layers))

        if !all_empty
            autoscale_buttons(plot)
            ig.SameLine()
            log_scale_buttons(plot)
            for L in plot.layers
                bottom_controls(L, plot)
            end
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

function var_combo(label, selected::Ref{Cint}, var_names, variable_data)
    n = length(var_names)
    preview = if n > 0
        name = var_names[selected[] + 1]
        type_label = var_type_label(variable_data[name])
        "$(name)  ($(type_label))"
    else
        ""
    end
    ig.SetNextItemWidth(250)

    changed = false
    if ig.BeginCombo(label, preview)
        for (i, name) in enumerate(var_names)
            if variable_data[name].type ∉ (VariableType_Scalar, VariableType_Vector)
                continue
            end

            is_selected = selected[] == i - 1
            if ig.Selectable(name, is_selected)
                selected[] = i - 1
                changed = true
            end

            ig.SameLine()

            ig.TextDisabled(var_type_label(variable_data[name]))
            if is_selected
                ig.SetItemDefaultFocus()
            end
        end

        ig.EndCombo()
    end

    return changed
end

# --- CorrelationLayer methods ---

function top_controls(layer::CorrelationLayer, ::Plot)
    client = state[].client
    variable_data = client.variable_data

    empty!(layer.variable_names)
    for (name, variable) in variable_data
        if variable.type in (VariableType_Scalar, VariableType_Vector)
            push!(layer.variable_names, name)
        end
    end
    sort!(layer.variable_names)

    n_variables = length(layer.variable_names)
    if n_variables > 0
        layer.x_var[] = clamp(layer.x_var[], 0, n_variables - 1)
        layer.y_var[] = clamp(layer.y_var[], 0, n_variables - 1)
    end

    id = layer.layer_id_str
    if ig.Button("Swap axes##$(id)")
        layer.x_var[], layer.y_var[] = layer.y_var[], layer.x_var[]
        swap!(layer.matcher)
        reverse!(layer.subscribed)
    end

    ig.SameLine()
    x_changed = var_combo("X##$(id)", layer.x_var, layer.variable_names, variable_data)
    ig.SameLine()
    y_changed = var_combo("Y##$(id)", layer.y_var, layer.variable_names, variable_data)

    if x_changed || y_changed
        empty!(layer.matcher)
    end
end

function prepare!(layer::CorrelationLayer, ::Plot, updated_variables)
    client = state[].client
    variable_data = client.variable_data
    n_variables = length(layer.variable_names)
    if n_variables == 0
        return Empty("No scalar or vector variables available to correlate.")
    end

    x_name = layer.variable_names[layer.x_var[] + 1]
    y_name = layer.variable_names[layer.y_var[] + 1]
    x = variable_data[x_name]
    y = variable_data[y_name]

    if x_name != layer.subscribed[1]
        unsubscribe_variable(state[], layer.subscribed[1])
        subscribe_variable(state[], x_name)
        layer.subscribed[1] = x_name
    end
    if y_name != layer.subscribed[2]
        unsubscribe_variable(state[], layer.subscribed[2])
        subscribe_variable(state[], y_name)
        layer.subscribed[2] = y_name
    end

    if x.type != y.type
        return Empty("Both variables must have the same type to correlate against each other.")
    end

    m = layer.matcher
    label = "$(x_name) vs $(y_name)"
    if x.type == VariableType_Scalar
        data_updated = ingest_scalar!(m, x, y, updated_variables, x_name, y_name)
        accu_changed = set_resolution!(m, layer.binning_resolution[])
        if data_updated || accu_changed
            if isnothing(m.accu)
                compute_fit!(layer.fit, m.y_data, m.x_data)
            else
                compute_fit!(layer.fit, m.accu.y_values, m.accu.x_values;
                             sigma=m.accu.sigma)
            end
        end
        if !isnothing(m.accu)
            return Band(m.accu.x_values, m.accu.y_lower, m.accu.y_upper,
                        m.accu.y_values, label)
        else
            return Line(m.x_data, m.y_data, label, :scatter)
        end
    elseif x.type == VariableType_Vector
        if ingest_vector!(m, x, y)
            compute_fit!(layer.fit, m.y_data, m.x_data)
        end
        if length(m.x_data) != length(m.y_data)
            return Empty("Cannot correlate vectors of different lengths ($(length(m.x_data)) vs $(length(m.y_data))).")
        end
        return Line(m.x_data, m.y_data, label, :scatter)
    else
        return Empty("Unsupported correlation of data type '$(x.type)'")
    end
end

function plot_frame!(::CorrelationLayer, frame::Line)
    ImPlot.PlotScatter(frame.label, frame.xs, frame.ys; spec=ImPlot.ImPlotSpec(; FillAlpha=0.5))
end

function plot_frame!(::CorrelationLayer, frame::Band)
    # Same label_id ties the band and line to one legend entry so ImPlot
    # gives them matching colors.
    ImPlot.PlotShaded(frame.label, frame.xs, frame.lower, frame.upper; spec=ImPlot.ImPlotSpec(; FillAlpha=0.5))
    ImPlot.PlotLine(frame.label, frame.xs, frame.line_ys)
end

draw_overlay(layer::CorrelationLayer, ::Plot, ::PlotType) = draw_fit_overlay(layer.fit)

function axis_labels(layer::CorrelationLayer)
    n = length(layer.variable_names)
    if n == 0
        return ("", "")
    end
    (layer.variable_names[layer.x_var[] + 1], layer.variable_names[layer.y_var[] + 1])
end

side_panel(layer::CorrelationLayer, ::Plot) = draw_fitting_settings(layer.layer_id_str, layer.fit)

function bottom_controls(layer::CorrelationLayer, ::Plot)
    n = length(layer.variable_names)
    if n == 0
        return
    end
    variable_data = state[].client.variable_data
    x_name = layer.variable_names[layer.x_var[] + 1]
    x = variable_data[x_name]
    if x.type != VariableType_Scalar
        return
    end
    id = layer.layer_id_str
    y_name = layer.variable_names[layer.y_var[] + 1]
    y = variable_data[y_name]

    ig.SameLine()
    if ig.Button("Clear##$(id)")
        clear_variable_data(x)
        clear_variable_data(y)
        clear_layer_data!(layer)
    end

    ig.SameLine()
    ig.SetNextItemWidth(120)
    ig.DragFloat("Binning resolution##$(id)",
                 layer.binning_resolution, 0.01f0,
                 0.0f0, typemax(Cfloat), "%.8f",
                 ig.ImGuiSliderFlags_AlwaysClamp)
end

# --- SpecLayer methods (struct defined above with the other layers) ---

# Latest spec for this layer, falling back to the last-seen one when the source
# variable is no longer advertising it.
function current_spec(layer::SpecLayer)
    store = get(state[].client.variable_data, layer.source_var, nothing)
    if !isnothing(store)
        idx = findfirst(s -> s.name == layer.spec_name, store.plot_specs)
        if !isnothing(idx)
            layer.last_spec = store.plot_specs[idx]
        end
    end
    return layer.last_spec
end

# Variables a spec references: each layer's primary `data`, plus any channel
# bound to a sibling variable (a String, as opposed to a Symbol dim).
function spec_variables(spec::PlotSpec)
    vars = Set{String}()
    for ls in spec.layers
        push!(vars, ls.data)
        for ch in (ls.x, ls.y, ls.color)
            if ch isa String
                push!(vars, ch)
            end
        end
    end
    return vars
end

# Subscribe to newly-referenced variables and drop ones no longer in the spec.
function reconcile_subscriptions!(layer::SpecLayer, spec::PlotSpec)
    desired = spec_variables(spec)
    for name in setdiff(desired, layer.subscribed)
        subscribe_variable(state[], name)
    end
    for name in setdiff(layer.subscribed, desired)
        unsubscribe_variable(state[], name)
    end
    empty!(layer.subscribed)
    union!(layer.subscribed, desired)
end

# Resolve x-axis values for a (possibly sliced) series. A String channel pulls
# from a sibling variable, a Symbol selects a dim of `data`, and nothing infers:
# a DimArray's first remaining dim lookup, else the integer index.
function resolve_x(ls::LayerSpec, data)
    if ls.x isa String
        other = get(state[].client.variable_data, ls.x, nothing)
        if !isnothing(other) && !(other.data isa ArrayMetadata)
            return other.data isa DimArray ? parent(other.data) : other.data
        end
        return 1:length(data)
    elseif ls.x isa Symbol && data isa DimArray && DD.hasdim(data, ls.x)
        return parent(lookup(data, ls.x))
    elseif data isa DimArray
        return parent(lookup(data)[1])
    else
        return 1:length(data)
    end
end

# Axis vectors for an image mark: a Symbol channel selects a dim's lookup,
# otherwise fall back to the store's axes (which may be nothing → pixel coords).
function image_axes(ls::LayerSpec, data, store)
    xax = (ls.x isa Symbol && data isa DimArray && DD.hasdim(data, ls.x)) ?
        parent(lookup(data, ls.x)) : store.x_axis
    yax = (ls.y isa Symbol && data isa DimArray && DD.hasdim(data, ls.y)) ?
        parent(lookup(data, ls.y)) : store.y_axis
    return xax, yax
end

function series_frame(ls::LayerSpec, xs, ys, label, color)
    if ls.mark === :bars
        bar_size = length(xs) > 1 ? Float64(abs(xs[2] - xs[1])) : 1.0
        return Bars(xs, ys, label, bar_size, false)
    else
        style = ls.mark === :scatter ? :scatter : :line
        return Line(xs, ys, label, style, color)
    end
end

# Append the frame(s) for one LayerSpec to `frames`.
function layerspec_frames!(layer::SpecLayer, ls::LayerSpec, updated_variables, frames::Vector{PlotType})
    store = get(state[].client.variable_data, ls.data, nothing)
    if isnothing(store) || store.data isa ArrayMetadata
        return
    end
    data = store.data
    if !(eltype(data) <: Real) || isempty(data)
        return
    end

    if ls.mark === :image || (data isa AbstractMatrix && !(ls.color isa Symbol))
        if isnothing(layer.image)
            layer.image = ImageState()
            layer.image.fixed_aspect[] = store.fixed_aspect
        end
        was_updated = haskey(updated_variables, ls.data)
        xax, yax = image_axes(ls, data, store)
        push!(frames, prepare_heatmap!(layer.image, data, xax, yax, was_updated))
        return
    end

    colordim = ls.color isa Symbol ? ls.color : nothing
    if !isnothing(colordim) && data isa DimArray && DD.hasdim(data, colordim)
        # coords are the grouping dim's lookup values (or its index range when
        # the dim has no explicit lookup), so the legend reads "<dim> - <value>".
        coords = parent(lookup(data, colordim))
        n = length(coords)
        for (i, slice) in enumerate(eachslice(data; dims = DD.dimnum(data, colordim)))
            # Always set an explicit colour (gradient → Viridis sample, otherwise
            # the discrete palette) so toggling `gradient` updates live: ImPlot
            # caches item->Color and only refreshes it when given a non-auto
            # colour, so handing back `nothing` would keep the stale colour.
            # Start at 0.2, not 0: Viridis near t=0 is almost black and vanishes
            # against the plot background.
            color = ls.gradient ?
                ImPlot.SampleColormap(Cfloat(0.2 + 0.8 * (i - 1) / max(n - 1, 1)), ImPlot.ImPlotColormap_Viridis) :
                ImPlot.GetColormapColor(i - 1)
            push!(frames, series_frame(ls, resolve_x(ls, slice), parent(slice), "$(colordim) - $(coords[i])", color))
        end
    else
        ys = data isa DimArray ? parent(data) : data
        label = isnothing(ls.label) ? ls.data : ls.label
        push!(frames, series_frame(ls, resolve_x(ls, data), ys, label, nothing))
    end
end

function prepare!(layer::SpecLayer, ::Plot, updated_variables)
    spec = current_spec(layer)
    if isnothing(spec)
        return Empty("Waiting for plot spec: $(layer.spec_name)")
    end
    reconcile_subscriptions!(layer, spec)

    frames = PlotType[]
    for ls in spec.layers
        layerspec_frames!(layer, ls, updated_variables, frames)
    end
    if isempty(frames)
        return Empty("Waiting for data: $(layer.spec_name)")
    end
    return frames
end

function axis_labels(layer::SpecLayer)
    spec = current_spec(layer)
    isnothing(spec) && return ("", "")
    return (something(spec.xlabel, ""), something(spec.ylabel, ""))
end

function window_title(layer::SpecLayer)
    spec = current_spec(layer)
    isnothing(spec) ? "" : something(spec.title, layer.spec_name)
end

function bottom_controls(layer::SpecLayer, ::Plot)
    if !isnothing(layer.image)
        image_controls(layer.image, layer.layer_id_str)
    end
end

function Base.close(layer::SpecLayer)
    if !isnothing(layer.image) && !isnothing(layer.image.gpu_heatmap)
        destroy!(layer.image.gpu_heatmap)
        layer.image.gpu_heatmap = nothing
    end
    for name in layer.subscribed
        unsubscribe_variable(state[], name)
    end
    empty!(layer.subscribed)
end
