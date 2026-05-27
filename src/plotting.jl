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

"""
Re-upload the 1D colormap texture if the active ImPlot colormap has changed.
Samples 256 points from the colormap and uploads as GL_RGBA8 with linear
filtering (smooth gradient between color stops).
"""
function update_colormap!(ctx::HeatmapContext, cmap::ImPlot.ImPlotColormap_)
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

# Per-plot fit configuration. Shared between Plot and CorrelationPlot so the
# side-panel fitting UI can be driven from a single struct.
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

# Re-run the selected fit against the plot's current X/Y samples. Called from
# the draw_plot data-update path so the popt stays in sync with what's shown.
function compute_fit!(fit::FitSettings, ydata::AbstractVector,
                      xdata::Maybe{AbstractVector}=nothing;
                      sigma::Maybe{AbstractVector}=nothing)
    name = FIT_TYPES[fit.fit_type[] + 1]
    t0 = time_ns()
    if name == "Line"
        fit.popt, fit.retcode = fit_line(ydata, xdata; sigma)
    elseif name == "Gaussian"
        fit.popt, fit.retcode = fit_gaussian(ydata, xdata; sigma)
    elseif name == "erf"
        fit.popt, fit.retcode = fit_erf(ydata, xdata; sigma)
    elseif name == "sin"
        fit.popt, fit.retcode = fit_sin(ydata, xdata; sigma)
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

@kwdef mutable struct Plot
    const name::String
    const id::String
    const open::Ref{Bool} = Ref(true)
    const autoscale_x::Ref{Bool} = Ref(true)
    const autoscale_y::Ref{Bool} = Ref(true)
    const log_x::Ref{Bool} = Ref(false)
    const log_y::Ref{Bool} = Ref(false)
    const fixed_aspect::Ref{Bool} = Ref(true)
    const show_side_panel::Ref{Bool} = Ref(false)
    const fit::FitSettings = FitSettings()
    const precision::Ref{Cint} = Ref(Cint(-1))

    # Colorbar interaction state. `clip_min`/`clip_max` are the values fed to
    # the colormap shader; `display_min`/`display_max` are the visible range
    # shown on the colorbar axis (>= clip range, controlled by mouse wheel).
    const autoscale_colorbar::Ref{Bool} = Ref(true)
    const log_scale::Ref{Bool} = Ref(false)
    const colorbar_clip_min::Ref{Cdouble} = Ref(0.0)
    const colorbar_clip_max::Ref{Cdouble} = Ref(1.0)
    const colorbar_display_min::Ref{Cdouble} = Ref(0.0)
    const colorbar_display_max::Ref{Cdouble} = Ref(1.0)
    colorbar_drag::Symbol = :none
    colorbar_display_zoomed::Bool = false

    gpu_heatmap::Union{Nothing, GPUHeatmap} = nothing
    dock_id::UInt32 = 0

    # ROI parameter values updated locally during a drag, keyed by parameter
    # name. Flushed to the engine when the user releases the mouse so we don't
    # flood it with per-frame updates.
    const pending_roi_updates::Dict{String, RectROI} = Dict{String, RectROI}()
end

Plot(name, counter::Int) = Plot(name, "$(name)##plot-$(counter)")

function Plot(name, id::String, dock_id = 0)
    subscriptions = state[].client.subscriptions
    precision = haskey(subscriptions, name) ? subscriptions[name].precision : -1
    plot = Plot(; name, id, dock_id=UInt32(dock_id), precision=Ref(Cint(precision)))
    subscribe_variable(state[], name; precision)
    plot
end

function Base.close(plot::Plot)
    if !isnothing(plot.gpu_heatmap)
        destroy!(plot.gpu_heatmap)
        plot.gpu_heatmap = nothing
    end

    unsubscribe_variable(state[], plot.name)
end

clear_plot(::Plot) = nothing

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
    if !compressed
        ig.BeginDisabled()
    end
    ig.SetNextItemWidth(120)
    if ig.InputInt("zfp precision##$(id)", precision)
        set_subscription_precision(state[], name, Int(precision[]))
    end
    if !compressed
        ig.EndDisabled()
    end
    if compressed
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
function interactive_colorbar(plot::Plot, size::ImVec2)
    display_min = plot.colorbar_display_min[]
    display_max = plot.colorbar_display_max[]
    clip_min = plot.colorbar_clip_min[]
    clip_max = plot.colorbar_clip_max[]
    # In log mode all four refs hold log10(value); the colorbar renders that
    # space directly and tick labels read as exponents.
    tick_format = plot.log_scale[] ? "1e%g" : "%g"

    # ColormapScale itself does not consume mouse input — without an overlay
    # button, clicks fall through to the parent window and start a window
    # move. Mark it as overlap-allowed and stack an InvisibleButton on top to
    # capture clicks/drags for the handles.
    ig.SetNextItemAllowOverlap()
    start_pos = ig.GetCursorScreenPos()
    ImPlot.ColormapScale("##colorbar_$(plot.id)",
                         display_min, display_max,
                         size, tick_format,
                         ImPlot.ImPlotColormapScaleFlags_None,
                         ImPlot.ImPlotColormap_Viridis)
    rect_min = ig.GetItemRectMin()
    rect_max = ig.GetItemRectMax()

    ig.SetCursorScreenPos(start_pos)
    ig.InvisibleButton("##colorbar_input_$(plot.id)",
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
        if active && plot.colorbar_drag !== :none
            near_handle = plot.colorbar_drag
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
        plot.colorbar_drag = d_min <= d_max ? :min : :max
    end

    if active && plot.colorbar_drag !== :none
        mouse_y = ig.GetMousePos().y
        new_v = y_to_value(mouse_y)
        eps = 1e-9 * max(abs(safe_span), 1.0)
        if plot.colorbar_drag === :min
            plot.colorbar_clip_min[] = min(new_v, plot.colorbar_clip_max[] - eps)
        else
            plot.colorbar_clip_max[] = max(new_v, plot.colorbar_clip_min[] + eps)
        end
        plot.autoscale_colorbar[] = false
        changed = true
    elseif !active
        plot.colorbar_drag = :none
    end

    if hovered && !active
        wheel = unsafe_load(ig.GetIO().MouseWheel)
        if wheel != 0
            mouse_y = ig.GetMousePos().y
            anchor = y_to_value(mouse_y)
            factor = wheel > 0 ? 0.85 : 1 / 0.85
            plot.colorbar_display_min[] = anchor + (display_min - anchor) * factor
            plot.colorbar_display_max[] = anchor + (display_max - anchor) * factor
            plot.colorbar_display_zoomed = true
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

# Draw any RectROI parameters associated with `plot.name` via @display as
# draggable rectangles on top of the image, with the parameter name labelled
# above the top-left corner. Sends a ChangeParameter when the user drags.
function draw_roi_overlays(plot::Plot, x_min, x_max, y_min, y_max)
    client = state[].client
    displays = get(client.context.displays, plot.name, String[])
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
        id = int32_hash(plot.id, param_name)
        held = Ref(false)

        if ImPlot.DragRect(id, x1, y1, x2, y2, rect_col, 0, C_NULL, C_NULL, held)
            xlo, xhi = minmax(x1[], x2[])
            ylo, yhi = minmax(y1[], y2[])
            new_roi = RectROI(xlo, ylo, xhi - xlo, yhi - ylo)
            if new_roi != param.value
                param.value = new_roi
                plot.pending_roi_updates[param_name] = new_roi
            end
        end

        if !held[] && haskey(plot.pending_roi_updates, param_name)
            client.pending_source_edit = param_name
            change_parameter(Parameter(param_name, plot.pending_roi_updates[param_name]))
            delete!(plot.pending_roi_updates, param_name)
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

# Draw a thin semi-transparent tab on the right edge of the plot region
# (`plot_min`/`plot_max` are screen-space corners). Renders via draw list + manual
# hit-testing so it doesn't perturb the parent's layout cursor. Toggles
# `plot.show_side_panel`.
function side_panel_tab(plot, plot_min::ImVec2, plot_max::ImVec2)
    tab_w = 14.0f0
    tab_h = 56.0f0
    x0 = plot_max.x - tab_w
    y0 = plot_min.y + (plot_max.y - plot_min.y - tab_h) / 2
    rect_min = ImVec2(x0, y0)
    rect_max = ImVec2(x0 + tab_w, y0 + tab_h)

    hovered = ig.IsMouseHoveringRect(rect_min, rect_max)
    active = hovered && ig.IsMouseDown(ig.ImGuiMouseButton_Left)
    clicked = hovered && ig.IsMouseClicked(ig.ImGuiMouseButton_Left)

    base = unsafe_load(ig.GetStyleColorVec4(ig.ImGuiCol_Button))
    alpha = active ? 1.0f0 : (hovered ? 0.75f0 : 0.35f0)
    col = ig.GetColorU32(ImVec4(base.x, base.y, base.z, alpha))

    draw_list = ig.GetWindowDrawList()
    rounding = unsafe_load(ig.GetStyle().FrameRounding)
    ig.AddRectFilled(draw_list, rect_min, rect_max, col, rounding)

    glyph = plot.show_side_panel[] ? ">" : "<"
    text_size = ig.CalcTextSize(glyph)
    text_pos = ImVec2(rect_min.x + (tab_w - text_size.x) / 2,
                      rect_min.y + (tab_h - text_size.y) / 2)
    text_col = ig.GetColorU32(ig.ImGuiCol_Text)
    ig.AddText(draw_list, text_pos, text_col, glyph)

    if clicked
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
        end

        name = FIT_TYPES[fit.fit_type[] + 1]
        names = fit_param_names(name)
        if name == "None"
            # nothing to show
        elseif !isnothing(fit.popt) && length(fit.popt) == length(names)
            flags = ig.ImGuiInputTextFlags_ReadOnly
            for (i, pname) in enumerate(names)
                # Re-fill the buffer each frame so the value stays in sync with
                # the latest fit (cheap — only a few short strings per panel).
                text = @sprintf("%g", fit.popt[i])
                buf = zeros(UInt8, 64)
                Util.strcpy!(buf, text)
                ig.SetNextItemWidth(150)
                ig.InputText("$(pname)##fit-$(id)", pointer(buf), length(buf), flags)
            end
            ig.TextDisabled(@sprintf("Fit time: %.2f ms", fit.elapsed * 1e3))
        elseif !isnothing(fit.retcode)
            ig.TextWrapped("Fit failed: $(fit.retcode)")
        end
    end
end

function draw_side_panel(plot::Plot, store, is_scalar::Bool, is_matrix::Bool)
    if !is_scalar && ig.CollapsingHeader("Compression##$(plot.id)")
        draw_compression_settings(plot.id, plot.name, plot.precision, store)
    end
    # Fitting only makes sense for 1D data.
    if is_matrix
        ig.BeginDisabled()
        ig.CollapsingHeader("Fitting##$(plot.id)")
        ig.EndDisabled()
    else
        draw_fitting_settings(plot.id, plot.fit)
    end
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

# Closes the plot-area child opened by begin_plot_area!. Draws the tab over the
# plot (when show_tab) so it captures clicks before being clipped, then if the
# panel is open draws it alongside via the caller-supplied `draw_panel`.
function end_plot_area!(plot, side_panel_width, plot_area_h, show_tab::Bool, draw_panel)
    if show_tab
        win_pos = ig.GetWindowPos()
        win_size = ig.GetWindowSize()
        side_panel_tab(plot, win_pos, ImVec2(win_pos.x + win_size.x, win_pos.y + win_size.y))
    end
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

function draw_plot(plot::Plot, store::Nothing, was_updated)
    ig.SetNextWindowSize((800, 500), ig.ImGuiCond_FirstUseEver)

    if ig.Begin(plot.id, plot.open)
        plot.dock_id = ig.GetWindowDockID()
        ig.Text("Waiting for data: $(plot.name)")
    end

    ig.End()
end

function draw_plot(plot::Plot, store, was_updated)
    ig.SetNextWindowSize((800, 500), ig.ImGuiCond_FirstUseEver)
    side_panel_width = 300f0

    data = store.data
    if ig.Begin("$(store.title)##$(plot.id)", plot.open)
        plot.dock_id = ig.GetWindowDockID()
        is_dimarray = data isa DimArray
        is_scalar = data isa CircularBuffer
        is_metadata = data isa ArrayMetadata
        label = store.title

        apply_autoscale(plot)

        no_data = is_metadata || length(data) == 0
        plot_size, plot_area_h = begin_plot_area!(plot, side_panel_width)
        if is_metadata
            ig.Text("Waiting for data: $(plot.name)")
        elseif no_data
            ig.Text("Array has length 0, nothing to plot")
        elseif data isa AbstractVector
            xs, ys = if is_scalar
                store.scalar_tids_cache, store.scalar_data_cache
            elseif !isnothing(store.x_axis)
                store.x_axis, data
            elseif is_dimarray
                parent(lookup(data)[1]), parent(data)
            else
                1:length(data), data
            end

            if was_updated
                compute_fit!(plot.fit, ys, xs)
            end

            if ImPlot.BeginPlot(store.title, store.xlabel, store.ylabel, plot_size)
                apply_log_scales(plot)
                if store.plot_type === :histogram
                    bar_size = length(xs) > 1 ? Float64(abs(xs[2] - xs[1])) : 1.0
                    ImPlot.PushStyleColor(ImPlot.ImPlotCol_Line, ig.ImVec4(0, 0, 0, 1))
                    ImPlot.PlotBars(label, xs, ys; bar_size)
                    ImPlot.PopStyleColor()
                elseif length(ys) == 1
                    ImPlot.PlotScatter(label, xs, ys)
                else
                    ImPlot.PlotLine(label, xs, ys)
                end
                draw_fit_overlay(plot.fit)
                draw_variable_overlays(plot.name)
                check_plot_interaction!(plot)
                ImPlot.EndPlot()
            end
        elseif data isa AbstractMatrix
            rows, cols = size(data)

            # Ensure GPU resources exist
            ctx = get_heatmap_context()
            needs_initial_upload = isnothing(plot.gpu_heatmap)
            if needs_initial_upload
                plot.gpu_heatmap = GPUHeatmap()
                plot.fixed_aspect[] = store.fixed_aspect
            end
            gpu = plot.gpu_heatmap

            # Update colormap if needed (use Viridis as default, index 4)
            update_colormap!(ctx, ImPlot.ImPlotColormap_Viridis)

            log = plot.log_scale[]
            log_changed = !needs_initial_upload && gpu.log_scale != log
            if was_updated || needs_initial_upload || log_changed
                if was_updated || needs_initial_upload
                    upload_data!(gpu, data)
                end
                if needs_initial_upload || log_changed || plot.autoscale_colorbar[]
                    dmin, dmax = sampled_pctile!(gpu.hist_buf, data, log)
                    plot.colorbar_clip_min[] = dmin
                    plot.colorbar_clip_max[] = dmax
                    # Don't stomp a manual zoom — only reset the visible range
                    # if the user has not adjusted it themselves (or just
                    # toggled log mode, which makes the old range meaningless).
                    if needs_initial_upload || log_changed || !plot.colorbar_display_zoomed
                        margin = 0.1 * (dmax - dmin)
                        plot.colorbar_display_min[] = dmin - margin
                        plot.colorbar_display_max[] = dmax + margin
                    end
                end
                render_colormapped!(gpu, ctx,
                                    plot.colorbar_clip_min[],
                                    plot.colorbar_clip_max[],
                                    log)
                gpu.log_scale = log
            end

            # Reserve space for the colorbar on the right
            colorbar_width = 100
            plot_width = max(plot_size.x - colorbar_width, 100)

            plot_flags = plot.fixed_aspect[] ? ImPlot.ImPlotFlags_Equal : ImPlot.ImPlotFlags_None
            if ImPlot.BeginPlot(store.title, ImVec2(plot_width, plot_size.y), plot_flags)
                ImPlot.SetupAxis(ImPlot.ImAxis_X1, store.xlabel)
                ImPlot.SetupAxis(ImPlot.ImAxis_Y1, store.ylabel, ImPlot.ImPlotAxisFlags_Invert)
                tex_ref = ig.ImTextureRef(ig.ImTextureID(gpu.output_tex))
                # Matplotlib convention: first dim = row (vertical, top→bottom),
                # second dim = col (horizontal, left→right). data[1,1] at plot
                # top-left; data[rows,cols] at plot bottom-right. Y axis is
                # inverted so y_min sits at the top — pass swapped y bounds so
                # the texture's data[1,:] row stays at the top.
                has_x_axis = !isnothing(store.x_axis)
                has_y_axis = !isnothing(store.y_axis)
                x_min = has_x_axis ? first(store.x_axis) : 0
                x_max = has_x_axis ? last(store.x_axis) : cols
                y_min = has_y_axis ? first(store.y_axis) : 0
                y_max = has_y_axis ? last(store.y_axis) : rows
                ImPlot.PlotImage("", tex_ref,
                                 ImPlot.ImPlotPoint(x_min, y_max),
                                 ImPlot.ImPlotPoint(x_max, y_min))

                draw_roi_overlays(plot, x_min, x_max, y_min, y_max)

                # Show pixel coordinates and intensity when hovering
                if ImPlot.IsPlotHovered()
                    mouse = ImPlot.GetPlotMousePos()
                    j = floor(Int, (mouse.x - x_min) / (x_max - x_min) * cols) + 1
                    i = floor(Int, (mouse.y - y_min) / (y_max - y_min) * rows) + 1
                    if 1 <= i <= rows && 1 <= j <= cols
                        val = data[i, j]
                        ImPlot.AnnotationClamped(mouse.x, mouse.y,
                                                 ImVec2(10, -10),
                                                 "[$i, $j] $val")
                    end
                end

                check_plot_interaction!(plot)
                ImPlot.EndPlot()
            end

            ig.SameLine()
            if interactive_colorbar(plot, ImVec2(colorbar_width, plot_size.y))
                render_colormapped!(gpu, ctx,
                                    plot.colorbar_clip_min[],
                                    plot.colorbar_clip_max[],
                                    plot.log_scale[])
            end
        end
        is_matrix = data isa AbstractMatrix
        end_plot_area!(plot, side_panel_width, plot_area_h, !no_data,
                       () -> draw_side_panel(plot, store, is_scalar, is_matrix))

        if !no_data
            autoscale_buttons(plot)

            if !(data isa AbstractMatrix)
                ig.SameLine()
                log_scale_buttons(plot)
            end

            if is_scalar
                ig.SameLine()
                if ig.Button("Clear##$(plot.id)")
                    clear_variable_data(store)
                end
            end

            if data isa AbstractMatrix
                ig.SameLine()
                ig.Checkbox("Fixed aspect", plot.fixed_aspect)
                ig.SameLine()
                if ig.Checkbox("Auto colorbar##$(plot.id)", plot.autoscale_colorbar)
                    if plot.autoscale_colorbar[]
                        plot.colorbar_display_zoomed = false
                    end
                end
                ig.SameLine()
                ig.Checkbox("Log##$(plot.id)", plot.log_scale)
            end
        end
    end

    ig.End()
end

# --- Correlation plot ---

@kwdef mutable struct CorrelationPlot
    const id::String
    const open::Ref{Bool} = Ref(true)
    const variable_names::Vector{String} = String[]
    const x_var::Ref{Cint} = Ref(Cint(0))
    const y_var::Ref{Cint} = Ref(Cint(0))
    const x_data::Vector{Float64} = Float64[]
    const y_data::Vector{Float64} = Float64[]
    const autoscale_x::Ref{Bool} = Ref(true)
    const autoscale_y::Ref{Bool} = Ref(true)
    const log_x::Ref{Bool} = Ref(false)
    const log_y::Ref{Bool} = Ref(false)
    const show_side_panel::Ref{Bool} = Ref(false)
    const fit::FitSettings = FitSettings()
    const subscribed::Vector{String} = ["", ""]
    # Motor-position binning resolution for scalar correlations. 0 disables
    # binning; >0 routes samples through an AccuPairSequence keyed on x.
    const binning_resolution::Ref{Cfloat} = Ref(Cfloat(0))

    accu::Maybe{AccuPairSequence} = nothing
    trainId::Int = -1
    dock_id::UInt32 = 0
end

function clear_plot(plot::CorrelationPlot)
    empty!(plot.x_data)
    empty!(plot.y_data)
    plot.accu = nothing
end

function CorrelationPlot(counter::Integer)
    CorrelationPlot(; id="CorrelationPlot##plot-$(counter)")
end

function CorrelationPlot(id::String, dock_id::Integer = 0)
    CorrelationPlot(; id, dock_id=UInt32(dock_id))
end

function Base.close(plot::CorrelationPlot)
    unsubscribe_variable(state[], plot.subscribed[1])
    unsubscribe_variable(state[], plot.subscribed[2])
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

function _var_combo(label, selected::Ref{Cint}, var_names, variable_data)
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

function swap_arrays(x, y)
    for i in eachindex(x, y)
        x[i], y[i] = y[i], x[i]
    end
end

function draw_plot(plot::CorrelationPlot, variable_data, updated_variables)
    # Update variable names
    empty!(plot.variable_names)
    for (name, variable) in variable_data
        if variable.type in (VariableType_Scalar, VariableType_Vector)
            push!(plot.variable_names, name)
        end
    end
    sort!(plot.variable_names)

    ig.SetNextWindowSize((800, 500), ig.ImGuiCond_FirstUseEver)

    # Clamp indices to valid range
    n_variables = length(plot.variable_names)
    if n_variables > 0
        plot.x_var[] = clamp(plot.x_var[], 0, n_variables - 1)
        plot.y_var[] = clamp(plot.y_var[], 0, n_variables - 1)
    end

    if ig.Begin(plot.id, plot.open)
        plot.dock_id = ig.GetWindowDockID()
        if ig.Button("Swap axes")
            plot.x_var[], plot.y_var[] = plot.y_var[], plot.x_var[]
            swap_arrays(plot.x_data, plot.y_data)
            reverse!(plot.subscribed)
            plot.accu = nothing
        end

        ig.SameLine()
        x_changed = _var_combo("X", plot.x_var, plot.variable_names, variable_data)
        ig.SameLine()
        y_changed = _var_combo("Y", plot.y_var, plot.variable_names, variable_data)

        if x_changed || y_changed
            empty!(plot.x_data)
            empty!(plot.y_data)
            plot.accu = nothing
        end

        side_panel_width = 300f0
        plot_size, plot_area_h = begin_plot_area!(plot, side_panel_width)

        # Resolved on each frame when there's a variable selection; reused by
        # the bottom-row controls after end_plot_area!.
        x = y = nothing
        x_name = y_name = ""
        types_match = false

        if n_variables > 0
            x_name = plot.variable_names[plot.x_var[] + 1]
            y_name = plot.variable_names[plot.y_var[] + 1]
            x = variable_data[x_name]
            y = variable_data[y_name]

            if x_name != plot.subscribed[1]
                unsubscribe_variable(state[], plot.subscribed[1])
                subscribe_variable(state[], x_name)
                plot.subscribed[1] = x_name
            end
            if y_name != plot.subscribed[2]
                unsubscribe_variable(state[], plot.subscribed[2])
                subscribe_variable(state[], y_name)
                plot.subscribed[2] = y_name
            end

            if x.type != y.type
                ig.Text("Both variables must have the same type to correlate against each other.")
            else
                types_match = true
                apply_autoscale(plot)

                if x.type == VariableType_Scalar
                    data_updated = false
                    if haskey(updated_variables, x_name) || haskey(updated_variables, y_name)
                        new_tids = get(updated_variables, x_name, Set{Int}())
                        if haskey(updated_variables, y_name)
                            union!(new_tids, updated_variables[y_name])
                        end

                        for tid in new_tids
                            xi = findfirst(==(tid), x.scalar_tids)
                            yi = findfirst(==(tid), y.scalar_tids)
                            if !isnothing(xi) && !isnothing(yi)
                                xv = x.data[xi]
                                yv = y.data[yi]
                                push!(plot.x_data, xv)
                                push!(plot.y_data, yv)
                                if !isnothing(plot.accu)
                                    append!(plot.accu, xv, yv)
                                end
                                data_updated = true
                            end
                        end
                    end

                    # Sync accu with the current resolution: rebuild from the
                    # raw sample history whenever the plot's resolution disagrees
                    # with what's stored in accu (covers initial creation,
                    # widget edits, swaps, and variable changes).
                    res = plot.binning_resolution[]
                    accu_changed = false
                    if res > 0 && (isnothing(plot.accu) || plot.accu.resolution != res)
                        plot.accu = AccuPairSequence(plot.x_data, plot.y_data, res)
                        accu_changed = true
                    elseif res <= 0 && !isnothing(plot.accu)
                        plot.accu = nothing
                        accu_changed = true
                    end

                    if data_updated || accu_changed
                        if isnothing(plot.accu)
                            compute_fit!(plot.fit, plot.y_data, plot.x_data)
                        else
                            compute_fit!(plot.fit, plot.accu.y_values, plot.accu.x_values;
                                         sigma=plot.accu.sigma)
                        end
                    end

                    if ImPlot.BeginPlot(plot.id, x_name, y_name, plot_size)
                        apply_log_scales(plot)
                        label = "$(x_name) vs $(y_name)"
                        if !isnothing(plot.accu)
                            # Same label_id ties the band and line to one
                            # legend entry, so ImPlot gives them matching
                            # colors.
                            ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_FillAlpha, 0.5)
                            ImPlot.PlotShaded(label, plot.accu.x_values,
                                              plot.accu.y_lower, plot.accu.y_upper)
                            ImPlot.PopStyleVar()
                            ImPlot.PlotLine(label, plot.accu.x_values, plot.accu.y_values)
                        else
                            ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_FillAlpha, 0.5)
                            ImPlot.PlotScatter(label, plot.x_data, plot.y_data)
                            ImPlot.PopStyleVar()
                        end
                        draw_fit_overlay(plot.fit)
                        check_plot_interaction!(plot)
                        ImPlot.EndPlot()
                    end
                elseif x.type == VariableType_Vector
                    # Only update both buffers together when both variables have
                    # data from the same train.
                    needs_copy = x.type == VariableType_Vector && x.trainId == y.trainId && x.trainId != plot.trainId
                    if needs_copy
                        resize!(plot.x_data, length(x.data))
                        resize!(plot.y_data, length(y.data))
                        copyto!(plot.x_data, x.data)
                        copyto!(plot.y_data, y.data)
                        plot.trainId = x.trainId
                        compute_fit!(plot.fit, plot.y_data, plot.x_data)
                    end

                    if ImPlot.BeginPlot(plot.id, x_name, y_name, plot_size)
                        apply_log_scales(plot)
                        ImPlot.PushStyleVar(ImPlot.ImPlotStyleVar_FillAlpha, 0.5)
                        ImPlot.PlotScatter("$(x_name) vs $(y_name)", plot.x_data, plot.y_data)
                        ImPlot.PopStyleVar()
                        draw_fit_overlay(plot.fit)
                        check_plot_interaction!(plot)
                        ImPlot.EndPlot()
                    end
                else
                    ig.Text("Unsupported correlation of data type '$(x.type)'")
                end
            end
        end

        end_plot_area!(plot, side_panel_width, plot_area_h, n_variables > 0,
                       () -> draw_side_panel(plot))

        if types_match
            autoscale_buttons(plot)
            ig.SameLine()
            log_scale_buttons(plot)

            if x.type == VariableType_Scalar
                ig.SameLine()
                if ig.Button("Clear##$(plot.id)")
                    clear_variable_data(x)
                    clear_variable_data(y)
                    clear_plot(plot)
                end

                ig.SameLine()
                ig.SetNextItemWidth(120)
                ig.DragFloat("Binning resolution##$(plot.id)",
                             plot.binning_resolution, 0.01f0,
                             0.0f0, typemax(Cfloat), "%.8f",
                             ig.ImGuiSliderFlags_AlwaysClamp)
            end
        end
    end

    ig.End()
end

function draw_side_panel(plot::CorrelationPlot)
    draw_fitting_settings(plot.id, plot.fit)
end
