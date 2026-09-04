## BinnedSequence

# Streaming scan binner: each value is folded into the bin selected by snapping
# its D coordinates onto per-axis levels. See scan_binning_design.md for the
# full design.
#
# Running statistics live in dense arrays indexed by sorted level position. A new
# level reallocates the arrays with a NaN/zero slice spliced in at its sorted
# position; new levels are rare and grids small, so this is cheap enough.
#
# `mean`/`count` are DimArrays whose position axes carry the (drifting) bin
# centres as lookups, rebuilt each append to track the drift.

# One binning axis, levels kept in ascending-anchor order. `anchors` are pinned
# at creation for nearest-neighbour lookup; `means` drift and are only reported
# as the bin position.
mutable struct BinAxis
    resolution::Float64
    anchors::Vector{Float64}
    means::Vector{Float64}
    counts::Vector{Int}
end

BinAxis(resolution::Real) = BinAxis(Float64(resolution), Float64[], Float64[], Int[])

nlevels(ax::BinAxis) = length(ax.anchors)

# Sorted position of the level within `resolution` of `x`, or 0 if `x` would
# open a new one.
function find_level(ax::BinAxis, x::Float64)
    p = searchsortedfirst(ax.anchors, x)
    best = 0
    best_dist = Inf
    for q in (p - 1, p)
        if 1 <= q <= length(ax.anchors)
            d = abs(x - ax.anchors[q])
            if d < best_dist
                best_dist = d
                best = q
            end
        end
    end
    return best_dist < ax.resolution ? best : 0
end

# Snap `x` to an existing level whose anchor is within `resolution`, folding `x`
# into that level's running mean; otherwise open a new level anchored at `x`.
# Returns (position, inserted): the level's sorted position and whether a new
# level was opened there. Anchors never drift, so revisits merge rather than
# spawning duplicates.
function snap!(ax::BinAxis, x::Real)
    x = Float64(x)
    best = find_level(ax, x)
    if best != 0
        ax.counts[best] += 1
        ax.means[best] += (x - ax.means[best]) / ax.counts[best]
        return (best, false)
    end

    p = searchsortedfirst(ax.anchors, x)
    insert!(ax.anchors, p, x)
    insert!(ax.means, p, x)
    insert!(ax.counts, p, 1)
    return (p, true)
end

# Streaming binner over D axes. `N` is the value's ndims (0 = scalar), `M = N + D`
# the ndims of `mean`. `mean`/`n` carry the value dims; the rest are per-bin. A
# mean element reads NaN until its first finite sample (`n == 0`). `elem_*` are a
# scalar variance aggregate over all finite elements, mirroring offline
# bin_by_steps' scalar std; `std` is the population std derived from them and
# updated per touched bin. `names`/`value_dims` label `mean`/`count`/`std`.
mutable struct BinnedSequence{N, D, M}
    axes::NTuple{D, BinAxis}
    names::NTuple{D, Symbol}
    value_dims::Tuple
    max_bins::Int
    value_size::NTuple{N, Int}
    nbins::Int                          # occupied bins, for length and the max_bins cap

    mean::DD.DimArray{Float64, M}       # (level1, ..., levelD, value dims...)
    count::DD.DimArray{Int, D}          # trains folded into each bin
    std::DD.DimArray{Float64, D}        # per-bin population std (NaN until a finite sample)
    n::Array{Int, M}                    # finite-sample count per element
    elem_sum::Array{Float64, D}
    elem_sumsq::Array{Float64, D}
    elem_n::Array{Int, D}
end

function BinnedSequence{N, D, M}(resolutions; names=(:x, :y), value_dims::Tuple=(),
                                 max_bins::Integer=10_000) where {N, D, M}
    if length(resolutions) != D
        throw(ArgumentError("expected $D resolutions, got $(length(resolutions))"))
    end
    axes = ntuple(d -> BinAxis(resolutions[d]), D)
    names = ntuple(d -> names[d], D)
    value_size = isempty(value_dims) ? ntuple(_ -> 0, N) : ntuple(i -> length(value_dims[i]), N)
    # Anonymous value dims as a placeholder until the first value is seen.
    vdims = isempty(value_dims) && N > 0 ?
        ntuple(i -> DD.Dim{Symbol(:dim_, i)}(Base.OneTo(value_size[i])), N) : value_dims

    posdims = ntuple(d -> DD.Dim{names[d]}(Float64[]), D)
    mean = DD.DimArray(Array{Float64, M}(undef, (ntuple(_ -> 0, D)..., value_size...)...), (posdims..., vdims...))
    count = DD.DimArray(zeros(Int, ntuple(_ -> 0, D)...), posdims)
    std = DD.DimArray(fill(NaN, ntuple(_ -> 0, D)...), posdims)
    BinnedSequence{N, D, M}(axes, names, vdims, Int(max_bins), value_size, 0, mean, count, std,
                            zeros(Int, (ntuple(_ -> 0, D)..., value_size...)...),
                            zeros(Float64, ntuple(_ -> 0, D)...),
                            zeros(Float64, ntuple(_ -> 0, D)...),
                            zeros(Int, ntuple(_ -> 0, D)...))
end

BinnedSequence{N, D}(resolutions; kwargs...) where {N, D} =
    BinnedSequence{N, D, N + D}(resolutions; kwargs...)

# Concrete aliases for the common cases (M = mean ndims = N + D).
const Scalar1dScan = BinnedSequence{0, 1, 1}
const Scalar2dScan = BinnedSequence{0, 2, 2}
const Vector1dScan = BinnedSequence{1, 1, 2}

Scalar1dScan(resolution::Float64; kwargs...) = Scalar1dScan([resolution]; kwargs...)
Scalar2dScan(res1::Float64, res2::Float64; kwargs...) = Scalar2dScan([res1, res2]; kwargs...)

Base.length(seq::BinnedSequence) = seq.nbins

# Sorted (ascending-position) bin-centre means for axis `d`.
positions(seq::BinnedSequence, d::Integer) = seq.axes[d].means

# Position dims labelled with the current (drifting) bin centres.
position_dims(seq::BinnedSequence{N, D}) where {N, D} =
    ntuple(d -> DD.Dim{seq.names[d]}(copy(seq.axes[d].means)), D)

# Recompute the stored population std for the bin at `key` from its aggregates.
function update_std!(seq::BinnedSequence{N, D}, key::NTuple{D, Int}) where {N, D}
    en = seq.elem_n[key...]
    if en >= 1
        mean = seq.elem_sum[key...] / en
        seq.std[key...] = sqrt(max(0.0, seq.elem_sumsq[key...] / en - mean^2))
    end
    return seq
end

# Refresh the position lookups on `mean`/`count`/`std` after the bin centres drift.
function relabel!(seq::BinnedSequence)
    seq.mean = DD.rebuild(seq.mean; dims=(position_dims(seq)..., seq.value_dims...))
    seq.count = DD.rebuild(seq.count; dims=position_dims(seq))
    seq.std = DD.rebuild(seq.std; dims=position_dims(seq))
    return seq
end

# Grow the dense arrays to the current level shape, splicing each axis's new
# level in at its sorted position (NaN/zero) and relocating the existing bins.
# `ins[d]` is the inserted sorted position on axis `d`, or 0 if none.
function regrid!(seq::BinnedSequence{N, D}, ins::NTuple{D, Int}) where {N, D}
    levels = ntuple(d -> nlevels(seq.axes[d]), D)
    vcolons = ntuple(_ -> Colon(), N)
    mean = fill(NaN, (levels..., seq.value_size...))
    n = zeros(Int, (levels..., seq.value_size...))
    count = zeros(Int, levels)
    std = fill(NaN, levels)
    elem_sum = zeros(Float64, levels)
    elem_sumsq = zeros(Float64, levels)
    elem_n = zeros(Int, levels)

    old_count = parent(seq.count)
    if !isempty(old_count)
        dest = ntuple(d -> [i < ins[d] || ins[d] == 0 ? i : i + 1 for i in 1:size(old_count, d)], D)
        mean[dest..., vcolons...] = parent(seq.mean)
        n[dest..., vcolons...] = seq.n
        count[dest...] = old_count
        std[dest...] = parent(seq.std)
        elem_sum[dest...] = seq.elem_sum
        elem_sumsq[dest...] = seq.elem_sumsq
        elem_n[dest...] = seq.elem_n
    end

    seq.mean = DD.DimArray(mean, (position_dims(seq)..., seq.value_dims...))
    seq.count = DD.DimArray(count, position_dims(seq))
    seq.std = DD.DimArray(std, position_dims(seq))
    seq.n = n
    seq.elem_sum = elem_sum
    seq.elem_sumsq = elem_sumsq
    seq.elem_n = elem_n
    return seq
end

function Base.append!(seq::BinnedSequence{N, D}, coords::NTuple{D, Real}, value) where {N, D}
    if !all(isfinite, coords) || (value isa Number && !isfinite(value))
        return seq
    end

    # Test the cap before snap! mutates the axes, so a rejected append doesn't
    # leave an empty level behind on the position axis.
    existing = ntuple(d -> find_level(seq.axes[d], Float64(coords[d])), D)
    fresh_bin = if any(==(0), existing)
        true
    else
        seq.count[existing...] == 0
    end
    if fresh_bin && seq.nbins >= seq.max_bins
        return seq
    end

    snapped = ntuple(d -> snap!(seq.axes[d], coords[d]), D)
    key = ntuple(d -> snapped[d][1], D)
    if N > 0 && all(==(0), seq.value_size)
        seq.value_size = size(value)
        seq.value_dims = ntuple(i -> DD.Dim{Symbol(:dim_, i)}(Base.OneTo(size(value, i))), N)
    end
    if size(seq.count) != ntuple(d -> nlevels(seq.axes[d]), D)
        regrid!(seq, ntuple(d -> snapped[d][2] ? snapped[d][1] : 0, D))
    else
        relabel!(seq)
    end

    if seq.count[key...] == 0
        seq.nbins += 1
    end
    push!(seq, key, value)
    return seq
end

Base.append!(seq::BinnedSequence{N, 1}, x::Real, value) where {N} = append!(seq, (x,), value)

# Fold `value`'s finite elements into the per-element running means/counts in
# `mean`/`n`, returning their (sum, sumsq, count) contribution to the variance
# aggregate. A scalar value folds through 0-dim views (eachindex yields one slot).
function fold_value!(mean, n, value)
    elem_sum = 0.0
    elem_sumsq = 0.0
    elem_n = 0
    for i in eachindex(mean)
        v = value[i]
        if isfinite(v)
            vf = Float64(v)
            c = (n[i] += 1)
            mean[i] = c == 1 ? vf : mean[i] + (vf - mean[i]) / c
            elem_sum += vf
            elem_sumsq += vf * vf
            elem_n += 1
        end
    end
    return elem_sum, elem_sumsq, elem_n
end

# Fold one value into the bin at `key`: a nan-aware per-element running mean plus
# the scalar variance aggregate. Non-finite elements are skipped (stay NaN until
# the first finite sample). DimArray parents are mutated in place.
function Base.push!(seq::BinnedSequence{N, D, M}, key::NTuple{D, Int}, value) where {N, D, M}
    vcolons = ntuple(_ -> Colon(), N)
    count = parent(seq.count)::Array{Int, D}
    count[key...] += 1
    mean = view(parent(seq.mean)::Array{Float64, M}, key..., vcolons...)
    n = view(seq.n, key..., vcolons...)

    elem_sum, elem_sumsq, elem_n = fold_value!(mean, n, value)
    seq.elem_sum[key...] += elem_sum
    seq.elem_sumsq[key...] += elem_sumsq
    seq.elem_n[key...] += elem_n
    update_std!(seq, key)
end

function reset!(seq::BinnedSequence{N, D, M}) where {N, D, M}
    for ax in seq.axes
        empty!(ax.anchors)
        empty!(ax.means)
        empty!(ax.counts)
    end

    seq.nbins = 0
    posdims = ntuple(d -> DD.Dim{seq.names[d]}(Float64[]), D)
    seq.mean = DD.DimArray(Array{Float64, M}(undef, (ntuple(_ -> 0, D)..., seq.value_size...)...),
                           (posdims..., seq.value_dims...))
    seq.count = DD.DimArray(zeros(Int, ntuple(_ -> 0, D)...), posdims)
    seq.std = DD.DimArray(fill(NaN, ntuple(_ -> 0, D)...), posdims)
    seq.n = zeros(Int, (ntuple(_ -> 0, D)..., seq.value_size...)...)
    seq.elem_sum = zeros(Float64, ntuple(_ -> 0, D)...)
    seq.elem_sumsq = zeros(Float64, ntuple(_ -> 0, D)...)
    seq.elem_n = zeros(Int, ntuple(_ -> 0, D)...)
end

## VectorHistory

# Ring buffer storing the history of fixed-size vector data. Each row is one
# vector of length `size`. It allocates double the requested capacity so
# `data()` returns a contiguous view without copying, shifting back to the front
# only when it fills up.
#
# With average_window > 1 each push! folds into the running average of the
# in-progress row, committing it only once average_window pushes accumulate.
mutable struct VectorHistory{T}
    const x::Matrix{T}        # (2 * max_len, size) backing store
    const size::Int           # length of each vector
    const max_len::Int        # logical capacity
    const average_window::Int # pushes folded into each committed row
    idx::Int                  # starting row (1-based)
    len::Int                  # number of committed vectors
    cur_count::Int            # pushes accumulated into the in-progress row
end

function VectorHistory{T}(size::Integer; max_len::Integer=1000,
                                 average_window::Integer=1) where {T}
    x = zeros(T, 2 * max_len, size)
    return VectorHistory{T}(x, size, max_len, average_window, 1, 0, 0)
end

function VectorHistory(data::AbstractVector; kwargs...)
    seq = VectorHistory(length(data); kwargs...)
    push!(seq, data)

    seq
end

VectorHistory(size::Integer; kwargs...) = VectorHistory{Float64}(size; kwargs...)

Base.length(s::VectorHistory) = s.len
capacity(s::VectorHistory) = s.max_len

# Contiguous view of all currently stored vectors, one per row.
data(s::VectorHistory) = @view s.x[s.idx:(s.idx + s.len - 1), :]

Base.getindex(s::VectorHistory, index...) = data(s)[index...]

function Base.push!(s::VectorHistory, item)
    if length(item) != s.size
        throw(ArgumentError("Item size $(length(item)) differs from the vector size $(s.size)!"))
    end

    # Fold the item into the in-progress row (the next, uncommitted slot). The
    # first push overwrites stale data; later ones update the running average.
    slot = @view s.x[s.idx + s.len, :]
    s.cur_count += 1
    if s.cur_count == 1
        slot .= item
    else
        slot .+= (item .- slot) ./ s.cur_count
    end

    # Commit the row once the averaging window is full, advancing the ring.
    if s.cur_count >= s.average_window
        s.cur_count = 0
        if s.len < s.max_len
            s.len += 1
        else
            s.idx += 1
            if s.idx == s.max_len + 1
                s.idx = 1
                s.x[1:s.max_len, :] .= s.x[(s.max_len + 1):end, :]
            end
        end
    end

    return s
end

function Base.append!(s::VectorHistory, items)
    for item in items
        push!(s, item)
    end

    return s
end

function reset!(s::VectorHistory)
    s.idx = 1
    s.len = 0
    s.cur_count = 0
    fill!(s.x, 0)

    return s
end

## ScanBinner

# Resolution may be given once (applied to every axis) or once per axis.
function resolve_resolutions(res, D)
    if length(res) == 1
        ntuple(_ -> Float64(res[1]), D)
    elseif length(res) == D
        ntuple(d -> Float64(res[d]), D)
    else
        throw(ArgumentError("Scan resolution has length $(length(res)) but the scan has $D axis(es)"))
    end
end

# A motor position is a scalar; tolerate a length-1 vector by taking its first.
scan_coord(x) = Float64(x isa AbstractArray ? first(x) : x)

# Only the three concrete binner shapes are supported.
function scan_binner(N, D, resolutions, value_dims; names=(:position1, :position2))
    if N == 0 && D == 1
        Scalar1dScan(resolutions...; names, value_dims)
    elseif N == 0 && D == 2
        Scalar2dScan(resolutions...; names, value_dims)
    elseif N == 1 && D == 1
        Vector1dScan(resolutions; names, value_dims)
    else
        throw(ArgumentError("scan supports scalar 1-D/2-D or vector 1-D values (got value ndims $N over $D axes)"))
    end
end

# Reusable streaming step-scan binner over `D` position axes, folding each `value`
# into the bin snapped from the latest motor position(s) (see BinnedSequence). Add
# values with `push!(sb, value, positions...)`; positions are sticky. When
# `keep_history` is true a bounded buffer retains recent (coords, value) so a
# resolution change can re-bin the tail; otherwise it resets the bins.
mutable struct ScanBinner
    const keep_history::Bool
    axis_names::NTuple{2, Symbol}

    seq::Union{Nothing, BinnedSequence}
    history::Union{Nothing, VectorHistory{Float64}}
    coord_history::Vector{CircularBuffer{Float64}}
    value_shape::Tuple{Vararg{Int}}
    value_dims::Tuple
    last_coords::Vector{Float64}
    coords_set::Vector{Bool}
    needs_rebin::Bool
end

function ScanBinner(D::Integer; keep_history::Bool=true, names=(:position1, :position2))
    ScanBinner(keep_history, (names[1], names[2]), nothing, nothing,
               CircularBuffer{Float64}[], (), (), fill(NaN, D), fill(false, D), false)
end

# Flag a resolution change to be applied on the next push!.
rebin!(sb::ScanBinner) = sb.needs_rebin = true

# Fold `value` into the bins, lazily building the binner on first value and
# replaying the tail on a pending rebin. Trailing `positions` update the sticky
# coordinates. A no-op until every axis has a position.
function Base.push!(sb::ScanBinner, value, positions...;
                    resolutions, history_budget::Integer=64 * 1024 * 1024)
    for (d, coord) in enumerate(positions)
        sb.last_coords[d] = scan_coord(coord)
        sb.coords_set[d] = true
    end
    if !all(sb.coords_set)
        return nothing
    end
    D = length(sb.last_coords)
    varr = value isa DD.AbstractDimArray ? parent(value) : value
    vshape = varr isa AbstractArray ? size(varr) : ()

    # A value of a different shape can't be folded into the existing bins or
    # recompute buffer, so start over.
    if !isnothing(sb.seq) && vshape != sb.value_shape
        @warn "Scan value shape changed from $(sb.value_shape) to $vshape, resetting the bins"
        sb.seq = nothing
        sb.history = nothing
        empty!(sb.coord_history)
        sb.needs_rebin = false
    end

    # Lazily build the binner and recompute buffer once the value shape is known.
    if isnothing(sb.seq)
        N = length(vshape)
        vlen = prod(vshape; init=1)
        sb.value_dims = value isa DD.AbstractDimArray ? DD.dims(value) :
                        varr isa AbstractArray ?
                            ntuple(i -> DD.Dim{Symbol(:dim_, i)}(Base.OneTo(size(varr, i))), ndims(varr)) : ()
        sb.seq = scan_binner(N, D, resolve_resolutions(resolutions, D), sb.value_dims; names=sb.axis_names)
        sb.value_shape = vshape
        if sb.keep_history
            max_len = clamp(history_budget ÷ (sizeof(Float64) * max(vlen, 1)), 1, 100_000)
            sb.history = VectorHistory{Float64}(vlen; max_len)
            sb.coord_history = [CircularBuffer{Float64}(max_len) for _ in 1:D]
        end
    end

    # On a resolution change, rebuild the bins by replaying the retained tail.
    if sb.needs_rebin
        N = length(sb.value_shape)
        sb.seq = scan_binner(N, D, resolve_resolutions(resolutions, D), sb.value_dims; names=sb.axis_names)
        if sb.keep_history
            rows = data(sb.history)
            for t in axes(rows, 1)
                coords = ntuple(d -> sb.coord_history[d][t], D)
                v = N == 0 ? rows[t, 1] : reshape(rows[t, :], sb.value_shape)
                append!(sb.seq, coords, v)
            end
            if length(sb.history) >= capacity(sb.history)
                @warn "Scan recompute buffer is full; a resolution change only re-bins the most recent $(capacity(sb.history)) trains" maxlog=1
            end
        end
        sb.needs_rebin = false
    end

    coords = ntuple(d -> sb.last_coords[d], D)
    if (varr isa Number && !isfinite(varr)) || !all(isfinite, coords)
        return nothing
    end

    # Fold this train into the bins and (optionally) the recompute buffer.
    if sb.keep_history
        push!(sb.history, varr isa AbstractArray ? vec(Float64.(varr)) : (Float64(varr),))
        for d in 1:D
            push!(sb.coord_history[d], coords[d])
        end
    end
    append!(sb.seq, coords, varr)
    return nothing
end
