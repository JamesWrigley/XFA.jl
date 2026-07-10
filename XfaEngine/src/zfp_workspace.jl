module ZfpWorkspaces

using ZfpCompression: zfp_compress!, zfp_decompress!, zfp_promote!, zfp_demote!
using NaNStatistics: nanmean
using Statistics: median!
using DimensionalData: DimensionalData as DD

export ZfpWorkspace, CompressedArray, compress_array,
       decompress_array, decompress_array!, allocate_array, restore_dims,
       should_compress, COMPRESSION_THRESHOLD

const COMPRESSION_THRESHOLD = 500

# zfp float fixed-accuracy mode: an absolute error tolerance, which (unlike
# fixed-precision's block-relative step) keeps faint signal alive and stops hot
# pixels smearing their neighbours. Integer inputs go via a float intermediate.
#
# k picks the tolerance: 0 = lossless; k > 0 = manual override, tol = k * noise
# sigma; k < 0 = adaptive default (auto_tol), value-range relative error.
#
# The default sets tol = max(range/n_levels, k_noise*noise): n_levels steps across
# the displayed range, but never finer than k_noise * the per-frame noise. Integer
# inputs on this path are always cameras (line plots are forced lossless upstream),
# shown only for monitoring, so they get an aggressive level count and a large
# k_noise that washes out incompressible sensor noise: a noise-only frame spans
# ~5-7 sigma, so k_noise ~20 pushes tol well past the range and collapses it toward
# its mean. Floats are derived/analysis data: fine level count and a small k_noise
# that only avoids wasting bits below the noise. The max() with range/n_levels is
# continuous, so structured frames (range/n_levels > k_noise*noise) keep their
# detail; the larger k_noise just treats more marginal frames as noise.
const N_LEVELS_INT = 6
const N_LEVELS_FLOAT = 64
const K_NOISE_INT = 20.0
const K_NOISE_FLOAT = 0.4

const LowBitInt = Union{Int8, UInt8, Int16, UInt16}
const ZfpNativeInt = Union{Int32, Int64}
const ZfpFloat = Union{Float32, Float64}
const Compressible = Union{LowBitInt, ZfpNativeInt, ZfpFloat}

# Float intermediate used to hand a value to zfp. Low-bit ints fit exactly in
# Float32 (max 65535 < 2^24); Int32/Int64 need Float64 (exact to 2^53).
zfp_float_type(::Type{<:LowBitInt}) = Float32
zfp_float_type(::Type{<:ZfpNativeInt}) = Float64
zfp_float_type(::Type{Float32}) = Float32
zfp_float_type(::Type{Float64}) = Float64

# Non-finite kind encoding stored in the per-element mask.
const KIND_FINITE = 0x00
const KIND_NAN    = 0x01
const KIND_POSINF = 0x02
const KIND_NEGINF = 0x03

# Reusable scratch buffers for ZFP (de)compression. A workspace is serial /
# single-owner: results from compress_array alias the workspace's internal
# buffers, so the caller must consume them before the next call.
@kwdef mutable struct ZfpWorkspace
    compressed::Vector{UInt8}        = UInt8[]    # main compressed payload
    mask_compressed::Vector{UInt8}   = UInt8[]    # compressed non-finite mask payload
    mask_kinds::Vector{UInt8}        = UInt8[]    # raw 0..3 non-finite mask
    mask_int32::Vector{Int32}        = Int32[]    # promoted-to-Int32 view of mask_kinds for ZFP
    float32_scratch::Vector{Float32} = Float32[]  # Float32 staging/restore buffer
    float64_scratch::Vector{Float64} = Float64[]  # Float64 staging/restore buffer
    noise_scratch::Vector{Float64}   = Float64[]  # sampled adjacent differences for the noise estimate
    value_scratch::Vector{Float64}   = Float64[]  # sampled values for the range estimate
end

# Info needed to rebuild a DimArray on the receiving side. Set when the input
# to compress_array was a DimArray; the underlying parent array is compressed
# as usual and these travel alongside. Dimension lookups are stored as-is
# (uncompressed).
struct DimArrayInfo
    dim_names::Vector{Symbol}
    dim_lookups::Vector{Any}
    name::String
    metadata::Dict
end

# Result of compress_array. `data` and `nonfinite_mask` alias the producing
# workspace's scratch buffers — copy them if you need to retain past the next
# compress_array call.
@kwdef struct CompressedArray
    data::Vector{UInt8}
    shape::Vector{Int}
    original_eltype::DataType
    nonfinite_mask::Union{Nothing, Vector{UInt8}} = nothing

    # Set when the input was a DimArray; carries the dimension info needed to
    # reconstruct it after decompression (see restore_dims).
    dims::Union{DimArrayInfo, Nothing} = nothing
end

# Adaptive default tolerance (negative k): quantize the robust displayed range
# into n_levels steps, but never finer than k_noise * the per-frame noise (see
# the constants above for the two regimes this floor serves).
function auto_tol(noise_sigma::Float64, signal_range::Float64, n_levels::Int, k_noise::Float64)
    return max(signal_range / n_levels, k_noise * noise_sigma)
end

# Fast, robust per-frame scale estimates from one subsampled pass over every 4th
# element. noise_sigma is the MAD of adjacent differences (differencing high-passes
# away the pedestal, /sqrt(2) undoes the variance doubling; 1.4826 is the Gaussian
# factor). signal_range is the 1st-99th percentile spread of the values — the
# robust displayed range, which unlike a central MAD tracks bimodal / HDR frames.
# Samples touching a non-finite value are skipped.
function estimate_scales(ws::ZfpWorkspace, arr::DenseArray{<:AbstractFloat})
    a = vec(arr)
    n = length(1:4:length(a) - 1)
    resize!(ws.noise_scratch, n)
    resize!(ws.value_scratch, n)
    m = 0
    @inbounds for i in 1:4:length(a) - 1
        x = Float64(a[i])
        d = Float64(a[i + 1]) - x
        if isfinite(d)
            m += 1
            ws.noise_scratch[m] = d
            ws.value_scratch[m] = x
        end
    end
    if m < 2
        return 0.0, 0.0
    end

    diffs = @view ws.noise_scratch[1:m]
    med = median!(diffs)
    @inbounds for j in eachindex(diffs)
        diffs[j] = abs(diffs[j] - med)
    end
    noise_sigma = 1.4826 * median!(diffs) / sqrt(2)

    vals = @view ws.value_scratch[1:m]
    lo = partialsort!(vals, clamp(round(Int, 0.01 * (m - 1)) + 1, 1, m))
    hi = partialsort!(vals, clamp(round(Int, 0.99 * (m - 1)) + 1, 1, m))
    signal_range = hi - lo

    return noise_sigma, signal_range
end

function should_compress(arr::AbstractArray)
    isa(arr, DenseArray) &&
        ndims(arr) in 1:4 &&
        length(arr) >= COMPRESSION_THRESHOLD &&
        eltype(arr) <: Compressible
end
should_compress(arr::DD.AbstractDimArray) = should_compress(parent(arr))
should_compress(_) = false

float_scratch(ws::ZfpWorkspace, ::Type{Float32}) = ws.float32_scratch
float_scratch(ws::ZfpWorkspace, ::Type{Float64}) = ws.float64_scratch

# Copy `src` into `dest`, replacing non-finite values with the local nanmean
# and recording each element's kind (finite / NaN / +Inf / -Inf) into `kinds`.
function sanitize_floats!(dest::DenseArray{T}, kinds::Vector{UInt8},
                          src::DenseArray{T}) where {T <: AbstractFloat}
    fill_value = nanmean(src)

    for i in eachindex(src)
        x = src[i]

        if isfinite(x)
            kinds[i] = KIND_FINITE
            dest[i] = x
        else
            kinds[i] = isnan(x) ? KIND_NAN : (x > 0 ? KIND_POSINF : KIND_NEGINF)
            dest[i] = isfinite(fill_value) ? fill_value : zero(T)
        end
    end
end

# Compress + lossless-encode the non-finite kind mask into ws.mask_compressed.
function compress_mask!(ws::ZfpWorkspace)
    n = length(ws.mask_kinds)
    resize!(ws.mask_int32, n)
    zfp_promote!(ws.mask_int32, ws.mask_kinds)
    zfp_compress!(ws.mask_compressed, ws.mask_int32)  # reversible / lossless
    return ws.mask_compressed
end

# Stage an integer input as its float intermediate in the workspace scratch.
# No non-finites are possible, so there's no mask.
function stage_float_input(ws::ZfpWorkspace, arr::DenseArray{T}) where {T <: Integer}
    F = zfp_float_type(T)
    scratch = float_scratch(ws, F)
    resize!(scratch, length(arr))
    copyto!(scratch, arr)
    return reshape(scratch, size(arr)), nothing
end

# Stage a float input: zero-copy when all values are finite, otherwise sanitize
# into the float scratch and build the non-finite kind mask to ship alongside.
function stage_float_input(ws::ZfpWorkspace, arr::DenseArray{T}) where {T <: ZfpFloat}
    if any(!isfinite, arr)
        scratch = float_scratch(ws, T)
        resize!(scratch, length(arr))
        sanitized = reshape(scratch, size(arr))
        resize!(ws.mask_kinds, length(arr))
        sanitize_floats!(sanitized, ws.mask_kinds, arr)
        return sanitized, compress_mask!(ws)
    else
        return arr, nothing
    end
end

# Fixed-accuracy compression; k = 0 is lossless, k < 0 the adaptive default, k > 0
# a manual override (see the k semantics up top). Integer inputs go through a
# float intermediate (see stage_float_input) and are rounded back on decompression.
function compress_array(ws::ZfpWorkspace, arr::DenseArray{T}; k::Real=-1) where {T <: Compressible}
    shape = collect(size(arr))
    src, mask = stage_float_input(ws, arr)

    if k == 0
        zfp_compress!(ws.compressed, src)  # lossless
    else
        noise_sigma, signal_range = estimate_scales(ws, src)
        n_levels, k_noise = T <: Integer ? (N_LEVELS_INT, K_NOISE_INT) : (N_LEVELS_FLOAT, K_NOISE_FLOAT)
        tol = k < 0 ? auto_tol(noise_sigma, signal_range, n_levels, k_noise) : k * noise_sigma
        zfp_compress!(ws.compressed, src; tol)
    end
    return CompressedArray(; data=ws.compressed, shape, original_eltype=T, nonfinite_mask=mask)
end

function compress_array(ws::ZfpWorkspace, arr::DD.AbstractDimArray; k::Real=-1)
    md = DD.metadata(arr)
    metadata = if md isa Dict
        md
    elseif md isa DD.NoMetadata
        Dict()
    else
        @error "Dropping DimArray metadata of unsupported type $(typeof(md)); expected Dict"
        Dict()
    end

    ca = compress_array(ws, parent(arr); k)
    ds = DD.dims(arr)
    info = DimArrayInfo(Symbol[DD.name(d) for d in ds],
                        Any[DD.lookup(d) for d in ds],
                        string(DD.name(arr)), metadata)
    return CompressedArray(; ca.data, ca.shape, ca.original_eltype, ca.nonfinite_mask, dims=info)
end

# Restore the non-finite values into `out` using the compressed kind mask.
function restore_nonfinite!(ws::ZfpWorkspace, out::DenseArray{T},
                            mask_bytes::Vector{UInt8}) where {T <: AbstractFloat}
    n = length(out)
    resize!(ws.mask_int32, n)
    zfp_decompress!(ws.mask_int32, mask_bytes)
    # zfp_promote! shifts/centers values rather than casting, so the Int32s
    # have to be demoted back to UInt8 to recover the kind codes.
    resize!(ws.mask_kinds, n)
    zfp_demote!(ws.mask_kinds, ws.mask_int32)

    for i in eachindex(out)
        k = ws.mask_kinds[i]
        if k == KIND_NAN
            out[i] = T(NaN)
        elseif k == KIND_POSINF
            out[i] = T(Inf)
        elseif k == KIND_NEGINF
            out[i] = T(-Inf)
        end
    end
end

# Allocate an uninitialized array with the right eltype and shape to receive
# the decompressed contents of `ca`. Pair with decompress_array!.
function allocate_array(ca::CompressedArray)
    return Array{ca.original_eltype}(undef, ca.shape...)
end

# Decompress `ca` into `out`. `out` must have the eltype and shape returned
# by `allocate_array(ca)`. Returns `out`.
function decompress_array!(ws::ZfpWorkspace, out::DenseArray{T},
                           ca::CompressedArray) where {T <: Compressible}
    if T !== ca.original_eltype
        throw(ArgumentError("eltype mismatch: out is $T, expected $(ca.original_eltype)"))
    end
    if collect(size(out)) != ca.shape
        throw(DimensionMismatch("size(out) = $(size(out)), expected $(ca.shape)"))
    end

    if T <: ZfpFloat
        zfp_decompress!(out, ca.data)
        if !isnothing(ca.nonfinite_mask)
            restore_nonfinite!(ws, out, ca.nonfinite_mask)
        end
    else
        # Integer original: decompress the float payload, then round back into
        # the integer type, clamping so lossy values near the type bounds don't
        # overflow the conversion.
        F = zfp_float_type(T)
        scratch = float_scratch(ws, F)
        resize!(scratch, length(out))
        fbuf = reshape(scratch, size(out))
        zfp_decompress!(fbuf, ca.data)
        lo, hi = F(typemin(T)), F(typemax(T))
        @inbounds for i in eachindex(out)
            out[i] = round(T, clamp(fbuf[i], lo, hi))
        end
    end
    return out
end

# Wrap a decompressed array as a DimArray when `ca` carried dimension info,
# rebuilding the dimensions from their stored names and lookups. Otherwise
# returns `arr` unchanged.
function restore_dims(arr::AbstractArray, ca::CompressedArray)
    if isnothing(ca.dims)
        return arr
    else
        info = ca.dims
        ds = Tuple(DD.rebuild(DD.name2dim(n), l) for (n, l) in zip(info.dim_names, info.dim_lookups))
        return DD.DimArray(arr, ds; name=info.name, metadata=info.metadata)
    end
end

# Convenience: allocate + decompress in one call. Reconstructs a DimArray when
# `ca` carried dimension info.
function decompress_array(ws::ZfpWorkspace, ca::CompressedArray)
    return restore_dims(decompress_array!(ws, allocate_array(ca), ca), ca)
end

end # module
