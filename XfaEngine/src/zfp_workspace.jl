module ZfpWorkspaces

using ZfpCompression: zfp_compress!, zfp_decompress!, zfp_promote!, zfp_demote!,
    zfp_clamp_int!, zfp_max_magnitude
using NaNStatistics: nanmean
using DimensionalData: DimensionalData as DD

export ZfpWorkspace, CompressedArray, compress_array,
       decompress_array, decompress_array!, allocate_array, restore_dims,
       should_compress, COMPRESSION_THRESHOLD

const COMPRESSION_THRESHOLD = 500

default_precision(::Type{<:Integer}) = 11
default_precision(::Type{<:AbstractFloat}) = 7
const NONFINITE_RADIUS = 5  # half-window: ~10 surrounding elements total

const LowBitInt = Union{Int8, UInt8, Int16, UInt16}
const ZfpNativeInt = Union{Int32, Int64}
const ZfpFloat = Union{Float32, Float64}
const Compressible = Union{LowBitInt, ZfpNativeInt, ZfpFloat}

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
    int32_scratch::Vector{Int32}     = Int32[]    # promoted low-bit ints / clamped Int32 input
    int64_scratch::Vector{Int64}     = Int64[]    # clamped Int64 input
    mask_kinds::Vector{UInt8}        = UInt8[]    # raw 0..3 non-finite mask
    mask_int32::Vector{Int32}        = Int32[]    # promoted-to-Int32 view of mask_kinds for ZFP
    float32_scratch::Vector{Float32} = Float32[]  # NaN/Inf-replaced copy of a Float32 input
    float64_scratch::Vector{Float64} = Float64[]  # NaN/Inf-replaced copy of a Float64 input
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
    promoted::Bool = false
    nonfinite_mask::Union{Nothing, Vector{UInt8}} = nothing

    # Set when the input was outside zfp's safe integer range and was clamped
    # to fit. Informational only — receivers see the clamped values and don't
    # need to take any special action.
    clamped::Bool = false

    # Set when the input was a DimArray; carries the dimension info needed to
    # reconstruct it after decompression (see restore_dims).
    dims::Union{DimArrayInfo, Nothing} = nothing
end

# A negative precision means "use the engine default", so callers can forward
# a per-client setting through without resolving it themselves.
resolve_precision(precision::Integer, ::Type{T}) where {T} =
    precision < 0 ? default_precision(T) : Int(precision)

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

native_int_scratch(ws::ZfpWorkspace, ::Type{Int32}) = ws.int32_scratch
native_int_scratch(ws::ZfpWorkspace, ::Type{Int64}) = ws.int64_scratch

# Copy `src` into `dest`, replacing non-finite values with the local nanmean
# and recording each element's kind (finite / NaN / +Inf / -Inf) into `kinds`.
function sanitize_floats!(dest::DenseArray{T}, kinds::Vector{UInt8},
                          src::DenseArray{T}) where {T <: AbstractFloat}
    fill_value = nanmean(src)

    n = length(src)

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

# Low-bit ints: promote to Int32 via the workspace scratch, then compress.
# No non-finites possible. Promoted values fit comfortably inside zfp's safe
# range (UInt16's full range is well below 2^30) so no clamping is needed.
function compress_array(ws::ZfpWorkspace, arr::DenseArray{T};
                        precision::Integer=-1) where {T <: LowBitInt}
    precision = resolve_precision(precision, T)
    shape = collect(size(arr))
    resize!(ws.int32_scratch, length(arr))
    promoted = reshape(ws.int32_scratch, size(arr))
    zfp_promote!(promoted, arr)
    zfp_compress!(ws.compressed, promoted; precision)
    return CompressedArray(; data=ws.compressed, shape, original_eltype=T, promoted=true)
end

# Natively-supported integer types: zero-copy when all values are within
# zfp's safe range; otherwise stage a clamped copy in the workspace scratch.
function compress_array(ws::ZfpWorkspace, arr::DenseArray{T};
                        precision::Integer=-1) where {T <: ZfpNativeInt}
    precision = resolve_precision(precision, T)
    shape = collect(size(arr))
    mag = zfp_max_magnitude(T)
    lo, hi = extrema(arr)

    clamped = false
    if lo < -mag || hi > mag
        scratch = native_int_scratch(ws, T)
        resize!(scratch, length(arr))
        copyto!(scratch, arr)
        clamped = zfp_clamp_int!(scratch)
        arr = reshape(scratch, size(arr))
    end

    zfp_compress!(ws.compressed, arr; precision)
    return CompressedArray(; data=ws.compressed, shape, original_eltype=T, clamped)
end

# Floats: zero-copy when all values are finite; otherwise sanitize into a
# float scratch and ship the kind mask alongside.
function compress_array(ws::ZfpWorkspace, arr::DenseArray{T};
                        precision::Integer=-1) where {T <: ZfpFloat}
    precision = resolve_precision(precision, T)
    shape = collect(size(arr))
    n = length(arr)

    mask = nothing
    if any(!isfinite, arr)
        scratch = float_scratch(ws, T)
        resize!(scratch, n)
        sanitized = reshape(scratch, size(arr))
        resize!(ws.mask_kinds, n)
        sanitize_floats!(sanitized, ws.mask_kinds, arr)

        arr = sanitized
        mask = compress_mask!(ws)
    end

    zfp_compress!(ws.compressed, arr; precision)
    return CompressedArray(; data=ws.compressed, shape, original_eltype=T, nonfinite_mask=mask)
end

function compress_array(ws::ZfpWorkspace, arr::DD.AbstractDimArray; precision::Integer=-1)
    md = DD.metadata(arr)
    metadata = if md isa Dict
        md
    elseif md isa DD.NoMetadata
        Dict()
    else
        @error "Dropping DimArray metadata of unsupported type $(typeof(md)); expected Dict"
        Dict()
    end

    ca = compress_array(ws, parent(arr); precision)
    ds = DD.dims(arr)
    info = DimArrayInfo(Symbol[DD.name(d) for d in ds],
                        Any[DD.lookup(d) for d in ds],
                        string(DD.name(arr)), metadata)
    return CompressedArray(; ca.data, ca.shape, ca.original_eltype, ca.promoted,
                           ca.nonfinite_mask, ca.clamped, dims=info)
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

    if ca.promoted
        resize!(ws.int32_scratch, length(out))
        intermediate = reshape(ws.int32_scratch, ca.shape...)
        zfp_decompress!(intermediate, ca.data)
        zfp_demote!(out, intermediate)
    else
        zfp_decompress!(out, ca.data)
        if !isnothing(ca.nonfinite_mask)
            restore_nonfinite!(ws, out, ca.nonfinite_mask)
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
