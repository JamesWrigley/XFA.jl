@Group struct MockInput end

update_sources(::MockInput, _) = nothing

struct KaraboDevice
    topic::String
    name::String
end

# Parse a KaraboDevice from a string, which may contain a topic prefix
# (e.g. "TOPIC//device_name").
function KaraboDevice(str::AbstractString)
    m = match(Context.topic_prefix_re, str)
    if !isnothing(m)
        return KaraboDevice(m.captures[1], m.captures[2])
    end
    return KaraboDevice("", str)
end

## Correlation group

@Group mutable struct Correlation
    buffer_size::Parameter{Int} = Parameter(update_buffer_size, 10_000)
    nbins::Parameter{Int} = Parameter(invalidate_histogram, 100)
    pulses::Parameter{Vector{Int}} = Parameter(invalidate_histogram, Int[])
    x::Parameter{Dependency}
    y::Parameter{Dependency}

    histogram::Hist2D{Float64} = Hist2D(; binedges=(1:10, 1:10))
    rebuild_histogram::Bool = true
    x_buffers::Vector{CircularBuffer{Float64}} = CircularBuffer{Float64}[]
    y_buffers::Vector{CircularBuffer{Float64}} = CircularBuffer{Float64}[]
    last_edge_update::Float64 = 0.0
end

function Base.show(io::IO, corr::Correlation)
    x = corr.x[]
    y = corr.y[]
    print(io, Correlation, "(x=$(x), y=$(y))")
end

function update_buffer_size(corr::Correlation, value)
    for buf in corr.x_buffers
        resize!(buf, value)
    end
    for buf in corr.y_buffers
        resize!(buf, value)
    end
    invalidate_histogram(corr, nothing)
end

function invalidate_histogram(corr::Correlation, _)
    corr.rebuild_histogram = true
end

function compute_edges(samples, nbins)
    lo, hi = floatmax(), floatmin()
    any_seen = false
    for x in samples
        if !isfinite(x)
            continue
        end
        any_seen = true
        lo = min(lo, x)
        hi = max(hi, x)
    end
    if !any_seen
        return range(-1.0, 1.0; length=nbins + 1)
    end
    if lo == hi
        lo -= 1
        hi += 1
    end
    return range(lo, hi; length=nbins + 1)
end

# Flatten the active pulses of a per-pulse buffer list into a single iterator
# of samples, for feeding into compute_edges.
function pulse_samples(buffers::Vector{CircularBuffer{Float64}}, pulses)
    idxs = isempty(pulses) ? eachindex(buffers) : pulses
    Iterators.flatten(buffers[i] for i in idxs if i <= length(buffers))
end

function build_histogram(corr::Correlation)
    binedges = (compute_edges(pulse_samples(corr.x_buffers, corr.pulses[]), corr.nbins[]),
                compute_edges(pulse_samples(corr.y_buffers, corr.pulses[]), corr.nbins[]))
    corr.histogram = Hist2D(; binedges, overflow=true)

    pulses = isempty(corr.pulses[]) ? eachindex(corr.x_buffers) : corr.pulses[]
    for i in pulses
        for (x, y) in zip(corr.x_buffers[i], corr.y_buffers[i])
            push!(corr.histogram, x, y)
        end
    end

    corr.rebuild_histogram = false
    corr.last_edge_update = time()
end

# Correlate two vector-valued variables by accumulating their points in
# per-pulse circular buffers and producing a 2D histogram.
@Variable function correlate(corr::Correlation, x -> Correlation.x, y -> Correlation.y)
    x_num = x isa Number
    y_num = y isa Number
    x_vec = x isa AbstractVector
    y_vec = y isa AbstractVector
    if !((x_num && y_num) || (x_vec && y_vec))
        return
    end

    # Figure out how many pulses we're working with
    n_pulses = 1
    if x_vec
        n_pulses = min(length(x), length(y))
    end

    # Adjust the internal buffers to the number of pulses
    if length(corr.x_buffers) < n_pulses
        while length(corr.x_buffers) < n_pulses
            push!(corr.x_buffers, CircularBuffer{Float64}(corr.buffer_size[]))
            push!(corr.y_buffers, CircularBuffer{Float64}(corr.buffer_size[]))
        end
    elseif length(corr.x_buffers) > n_pulses
        while length(corr.x_buffers) > n_pulses
            pop!(corr.x_buffers)
            pop!(corr.y_buffers)
        end
    end

    # Rebuild the histogram if necessary
    old_bins = time() - corr.last_edge_update >= 5
    if !isempty(corr.x_buffers)
        few_bins = length(corr.x_buffers[1]) < 20

        if corr.rebuild_histogram || old_bins || few_bins
            build_histogram(corr)
        end
    end

    for i in 1:n_pulses
        push!(corr.x_buffers[i], x[i])
        push!(corr.y_buffers[i], y[i])

        if isempty(corr.pulses[]) || i ∈ corr.pulses[]
            push!(corr.histogram, x[i], y[i])
        end
    end

    # Transform histogram weights (indexed [x_bin, y_bin]) into image-style
    # layout: first dim = row (top→bottom with y_max at top), second dim = col
    # (left→right with x_max at right).
    xe, ye = binedges(corr.histogram)
    data = permutedims(bincounts(corr.histogram))
    return VariableData(; data,
                        x_axis=collect(xe), y_axis=collect(ye),
                        xlabel=corr.x.value.name, ylabel=corr.y.value.name,
                        fixed_aspect=false, compress=false)
end

## Histogram1D

# Streaming 1D histogram with periodic edge rebinning. Usable both as a
# postprocessor (attach with `@postprocess Histogram1D(...)`) and standalone
# inside a variable body (append!() raw samples and read bincounts/bincenters).
# When `binedges` is empty the bin range is auto-fitted from the warm-up
# buffer using `nbins`; setting `binedges` pins the edges explicitly.
mutable struct Histogram1D <: AbstractPostprocessor
    nbins::Parameter{Int}
    binedges::Parameter{Tuple{Float64, Float64}}
    normalize::Parameter{Bool}
    windowed::Parameter{Bool}
    # Capacity expressed in trains; the underlying sample buffer is (re)sized
    # to `length(xs) * buffer_size` on each `append!` call.
    buffer_size::Parameter{Int}

    const buffer::CircularBuffer{Float64}
    histogram::Hist1D{Float64}
    rebuild::Bool
    last_edge_update::Float64
end

function invalidate_edges(h::Histogram1D, _)
    h.rebuild = true
end

# The `binedges` parameter is treated as auto-fitted whenever the user hasn't
# pinned it via the GUI. `rebuild!` then writes the discovered range back via
# `tryset` so the GUI shows live values, but `set_by_user` stays false so the
# next rebuild still re-fits from data.
auto_edges(binedges::Parameter{Tuple{Float64, Float64}}) = !binedges.set_by_user

function Histogram1D(; buffer_size::Integer=10, nbins::Integer=100,
                     binedges::Tuple{Real, Real}=(0.0, 0.0),
                     normalize::Bool=false, windowed::Bool=false)
    edges = (Float64(binedges[1]), Float64(binedges[2]))
    explicit = edges[1] < edges[2]
    initial = explicit ? range(edges[1], edges[2]; length=Int(nbins) + 1) :
                         range(-1.0, 1.0; length=Int(nbins) + 1)
    edge_param = Parameter("", edges, explicit, invalidate_edges, invalidate_edges)
    Histogram1D(Parameter(invalidate_edges, Int(nbins)),
                edge_param,
                Parameter(normalize),
                Parameter(invalidate_edges, windowed),
                Parameter(invalidate_edges, Int(buffer_size)),
                CircularBuffer{Float64}(buffer_size),
                Hist1D(; binedges=initial, overflow=true),
                true, 0.0)
end

default_name(::Histogram1D) = "histogram"

FHist.bincounts(h::Histogram1D) = bincounts(h.histogram)
FHist.bincenters(h::Histogram1D) = bincenters(h.histogram)

const HIST_REBIN_INTERVAL = 5.0

function rebuild!(h::Histogram1D)
    edges = if auto_edges(h.binedges)
        fitted = compute_edges(h.buffer, h.nbins[])
        # Surface the auto-fitted range back to the GUI without claiming the
        # binedges parameter — tryset is a no-op once the user pins a range.
        tryset(h.binedges, (Float64(first(fitted)), Float64(last(fitted))))
        fitted
    else
        lo, hi = h.binedges[]
        range(lo, hi; length=h.nbins[] + 1)
    end
    h.histogram = Hist1D(; binedges=edges, overflow=true)
    for x in h.buffer
        if isfinite(x)
            push!(h.histogram, x)
        end
    end
    h.rebuild = false
    h.last_edge_update = time()
end

function Base.append!(h::Histogram1D, xs)
    # In windowed mode the histogram is just a view of the circular buffer,
    # so size the buffer for the requested train count, feed it, and rebuild.
    if h.windowed[]
        desired_cap = length(xs) * h.buffer_size[]
        if desired_cap > 0 && h.buffer.capacity != desired_cap
            resize!(h.buffer, desired_cap)
        end
        append!(h.buffer, xs)
        rebuild!(h)
        return h
    end

    # The buffer only exists to settle auto-fitted bin edges during warm-up.
    # With explicit `binedges` we still feed it so a parameter change can
    # replay history through `rebuild!`, but skip the time/sparsity triggers
    # since the edges are pinned.
    explicit_edges = !auto_edges(h.binedges)
    needs_rebuild = if isfull(h.buffer)
        h.rebuild
    elseif explicit_edges
        append!(h.buffer, xs)
        h.rebuild
    else
        append!(h.buffer, xs)
        stale = time() - h.last_edge_update >= HIST_REBIN_INTERVAL
        sparse = !isempty(h.buffer) && length(h.buffer) < 20
        h.rebuild || sparse || stale || isfull(h.buffer)
    end

    if needs_rebuild
        rebuild!(h)
    else
        for v in xs
            if isfinite(v)
                push!(h.histogram, v)
            end
        end
    end
    return h
end

Base.push!(h::Histogram1D, x::Number) = append!(h, (x,))

# Postprocessor entry point: bin the variable's output and return self so
# `wrap_result` can package the current histogram state into a VariableData.
(h::Histogram1D)(data::Number) = (push!(h, data); h)
(h::Histogram1D)(data) = (append!(h, data); h)

# Lets user variables return a Histogram1D directly; the engine wraps it
# into a VariableData with bincounts as data, bincenters as x_axis, and the
# :histogram plot type so the GUI draws bars.
function VariableData(h::Histogram1D; kwargs...)
    counts = collect(Float64, bincounts(h))
    if h.normalize[]
        total = sum(counts)
        if total > 0
            counts ./= total
        end
    end
    return VariableData(; data=counts, x_axis=collect(bincenters(h)),
                        plot_type=:histogram, kwargs...)
end

## Reducer postprocessor

# Postprocessor that reduces a variable's data with `nanmean`. With an empty
# `dims` (the default) all dimensions are reduced to a scalar; otherwise the
# mean is taken along the given dims via nanmean's `dim` kwarg (which drops
# the reduced axes) using a buffer preallocated via `allocate_nanmean`. The
# buffer is reallocated when the input type or requested dims change.
@kwdef mutable struct Reducer{R, D, A} <: AbstractPostprocessor where {R, A}
    dims::Parameter{OptionalDims} = Parameter(OptionalDims())
    buffer::AbstractArray = []
    buffer_key::UInt = UInt(0)

    default_name::String
    reducer::R
    dims_reducer::D
    allocator::A
end

function Base.show(io::IO, r::Reducer)
    print(io, Reducer, "($(nameof(r.reducer)))")
end

default_name(r::Reducer) = r.default_name

Mean(; dims=()) = Reducer(; dims=Parameter(OptionalDims(isempty(dims) ? Int[] : Vector{Int}(collect(dims)))),
                          default_name="mean",
                          reducer=nanmean,
                          dims_reducer=nanmean!,
                          allocator=allocate_nanmean)

Sum(; dims=()) = Reducer(; dims=Parameter(OptionalDims(isempty(dims) ? Int[] : Vector{Int}(collect(dims)))),
                          default_name="sum",
                          reducer=nansum,
                          dims_reducer=nansum!,
                          allocator=allocate_nansum)

function (r::Reducer)(data)
    dims = Tuple(r.dims[].dims)
    if isempty(dims)
        return r.reducer(data)
    end

    key = hash((size(data), eltype(data), dims))
    if key != r.buffer_key
        r.buffer = r.allocator(data, dims)
        r.buffer_key = key
    end

    return r.dims_reducer(r.buffer, data; dim=dims)
end

## MovingAvg postprocessor

# Postprocessor that maintains an exponentially weighted moving average of the
# input array. The buffer is reallocated when the input's size or eltype
# changes; `nsamples` controls the EWMA window via alpha = 2/(nsamples+1).
mutable struct MovingAvg <: AbstractPostprocessor
    nsamples::Parameter{Int}
    buffer::AbstractArray
    buffer_key::UInt
end

MovingAvg(; nsamples::Integer=10) = MovingAvg(Parameter(Int(nsamples)), Float64[], UInt(0))

default_name(::MovingAvg) = "moving_avg"

function update_moving_avg!(buffer::AbstractArray{T}, data, nsamples) where {T}
    alpha = T(2 / (nsamples + 1))
    @. buffer = alpha * data + (1 - alpha) * buffer
    return buffer
end

# Center of mass of an array, weighting each entry by its value. For 1D
# returns a scalar coordinate; for 2D returns an (x, y) tuple where x is the
# column index and y is the row index. Non-finite entries are skipped.
function center_of_mass(data::AbstractVector)
    total = 0.0
    weighted = 0.0
    for i in eachindex(data)
        v = data[i]
        if isfinite(v)
            total += v
            weighted += i * v
        end
    end

    com = weighted / total
    return com in axes(data, 1) ? com : NaN
end

function center_of_mass(data::AbstractMatrix)
    total = zero(eltype(data))
    wx = zero(eltype(data))
    wy = zero(eltype(data))
    for j in axes(data, 2), i in axes(data, 1)
        v = data[i, j]
        if isfinite(v)
            total += v
            wx += j * v
            wy += i * v
        end
    end

    com_x = wx / total
    com_y = wy / total
    if !isfinite(com_x) || round(Int, com_x) ∉ axes(data, 2)
        com_x = NaN
    end
    if !isfinite(com_y) || round(Int, com_y) ∉ axes(data, 1)
        com_y = NaN
    end

    return (com_x, com_y)
end

function (m::MovingAvg)(data::AbstractArray)
    key = hash((size(data), eltype(data)))
    if key != m.buffer_key
        m.buffer = float(eltype(data)).(data)
        m.buffer_key = key
        return m.buffer
    end

    return update_moving_avg!(m.buffer, data, m.nsamples[])
end
