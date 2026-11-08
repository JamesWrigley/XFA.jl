@Group struct MockInput end

update_sources(::MockInput, _) = nothing

struct KaraboDevice
    topic::String
    name::String
end

# A source reported by an input (see get_sources()). `ambiguous` is a GUI-only
# field set by the client when the same source name is available in more than
# one topic.
struct SourceInfo
    topic::String
    name::String
    class_id::String
    ambiguous::Bool
end

SourceInfo(topic, name, class_id) = SourceInfo(topic, name, class_id, false)

# Parse a KaraboDevice from a string, which may contain a topic prefix
# (e.g. "TOPIC//device_name").
function KaraboDevice(str::AbstractString)
    m = match(Context.topic_prefix_re, str)
    if !isnothing(m)
        return KaraboDevice(String(m[1]), String(m[2]))
    end
    return KaraboDevice("", String(str))
end

# The only device of class `class` in `devices`, to fill in the parameter
# `param_name` (see `device_classes`). Errors if there are none or several, the
# parameter must then be set explicitly.
function unique_device(devices::Vector{SourceInfo}, class::AbstractString, param_name::AbstractString)
    matches = filter(dev -> dev.class_id == class, devices)
    if length(matches) == 1
        return KaraboDevice(only(matches).topic, only(matches).name)
    elseif isempty(matches)
        throw(Context.XfaContextException("No $(class) device found for '$(param_name)', it must be set explicitly"))
    else
        names = join(["$(dev.topic)//$(dev.name)" for dev in matches], ", ")
        throw(Context.XfaContextException("Found several $(class) devices for '$(param_name)' ($(names)), " *
                                          "it must be set explicitly"))
    end
end

## Correlation group

@Group mutable struct Correlation
    trains::Parameter{Int} = Parameter(resize_buffers, 10_000)
    nbins::Parameter{Int} = Parameter(invalidate_histogram, 100)
    pulses::Parameter{IndexSelection} = Parameter(invalidate_histogram, IndexSelection())
    fit::Parameter{Bool} = Parameter(false)
    clear::Callback = Callback("Clear", clear_buffers)
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

function resize_buffers(corr::Correlation, value)
    for buf in corr.x_buffers
        resize!(buf, value)
    end
    for buf in corr.y_buffers
        resize!(buf, value)
    end
    invalidate_histogram(corr, nothing)
end

function clear_buffers(corr::Correlation)
    foreach(empty!, corr.x_buffers)
    foreach(empty!, corr.y_buffers)
    invalidate_histogram(corr, nothing)
end

function invalidate_histogram(corr::Correlation, _)
    corr.rebuild_histogram = true
end

function compute_edges(samples, nbins)
    lo, hi = Inf, -Inf
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

# The selected pulses that are present in `buffers`, all of them if none are
# selected.
function active_pulses(buffers, pulses)
    isempty(pulses) ? eachindex(buffers) : intersect(pulses, eachindex(buffers))
end

# Flatten the active pulses of a per-pulse buffer list into a single iterator
# of samples, for feeding into compute_edges.
function pulse_samples(buffers::Vector{CircularBuffer{Float64}}, pulses)
    Iterators.flatten(buffers[i] for i in active_pulses(buffers, pulses))
end

function build_histogram(corr::Correlation)
    binedges = (compute_edges(pulse_samples(corr.x_buffers, corr.pulses[].indices), corr.nbins[]),
                compute_edges(pulse_samples(corr.y_buffers, corr.pulses[].indices), corr.nbins[]))
    corr.histogram = Hist2D(; binedges, overflow=true)

    for i in active_pulses(corr.x_buffers, corr.pulses[].indices)
        for (x, y) in zip(corr.x_buffers[i], corr.y_buffers[i])
            push!(corr.histogram, x, y)
        end
    end

    corr.rebuild_histogram = false
    corr.last_edge_update = time()
end

function linear_fit(corr::Correlation)
    pulses = active_pulses(corr.x_buffers, corr.pulses[].indices)
    xs = reduce(vcat, (corr.x_buffers[i] for i in pulses); init=Float64[])
    ys = reduce(vcat, (corr.y_buffers[i] for i in pulses); init=Float64[])

    if length(xs) < 2
        return nothing
    else
        sol = solve(CurveFitProblem(xs, ys), LinearCurveFitAlgorithm())
        if successful_retcode(sol.retcode)
            r2 = 1 - rss(sol) / sum(abs2, ys .- nanmean(ys))
            return [sol.u..., r2]
        else
            return nothing
        end
    end
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
            push!(corr.x_buffers, CircularBuffer{Float64}(corr.trains[]))
            push!(corr.y_buffers, CircularBuffer{Float64}(corr.trains[]))
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
        if !isfinite(x[i]) || !isfinite(y[i])
            continue
        end

        push!(corr.x_buffers[i], x[i])
        push!(corr.y_buffers[i], y[i])

        if isempty(corr.pulses[].indices) || i ∈ corr.pulses[].indices
            push!(corr.histogram, x[i], y[i])
        end
    end

    # [slope, intercept, R^2]
    fit = corr.fit[] ? linear_fit(corr) : nothing
    @add_subvariable("fit", fit)
    plot_specs = if isnothing(fit)
        PlotSpec[]
    else
        overlay = ModelOverlay(:linear; params="$(Meta.name[]).fit", title="linear fit", to_x_intercept=true)
        [PlotSpec("fit", [LayerSpec(; mark=:image)]; models=[overlay], fixed_aspect=false)]
    end

    # Transform histogram weights (indexed [x_bin, y_bin]) into image-style
    # layout: first dim = row (top→bottom with y_max at top), second dim = col
    # (left→right with x_max at right).
    xe, ye = binedges(corr.histogram)
    data = permutedims(bincounts(corr.histogram))
    return VariableData(; data,
                        x_axis=collect(xe), y_axis=collect(ye),
                        xlabel=corr.x.value.name, ylabel=corr.y.value.name,
                        fixed_aspect=false, compress=false, plot_specs)
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
    edge_param = Parameter(; value=edges, set_by_user=explicit,
                           update_handler=invalidate_edges, initializer=invalidate_edges)
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
    for i in eachindex(buffer, data)
        x = data[i]
        if !isfinite(x)
            continue
        elseif isfinite(buffer[i])
            buffer[i] = alpha * x + (1 - alpha) * buffer[i]
        else
            buffer[i] = x
        end
    end
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
        m.buffer = map(float(eltype(data)), data)
        m.buffer_key = key
        return m.buffer
    end

    return update_moving_avg!(m.buffer, data, m.nsamples[])
end

# The buffer holds plain values, so the output carries the latest input's lookups
(m::MovingAvg)(data::DD.AbstractDimArray) = DD.rebuild(data, m(parent(data)))

# Scalars go through the same buffer as a zero-dimensional array
(m::MovingAvg)(data::Number) = m(fill(data))[]

## Scan postprocessor

# Bins its source by the position of a scanned motor (see `ScanBinner`). In
# automatic mode (the default) it follows a Karabacon scantool: the scantool is
# picked from the inputs' devices, the motor from the scantool's configuration,
# each scan is binned afresh at half its step size, and the last `n_scans`
# finished scans are kept as records. In manual mode `motor` and `resolution`
# are set by hand and the bins accumulate.
mutable struct Scan <: AbstractPostprocessor
    automatic::Parameter{Bool}
    scantool::Parameter{KaraboDevice}
    motor::Parameter{Dependency}
    # The motors' position property. Karabacon's motor interfaces differ here
    # (e.g. slits use actualPositionX, monochromators actualEnergy).
    position_property::Parameter{String}
    resolution::Parameter{Float64}
    n_scans::Parameter{Int}
    history_budget::Parameter{Int}
    # For 1-D scans moving several motors together: the first of these that's
    # active is followed, otherwise the scantool's first motor.
    motor_priority::Vector{String}

    binner::Union{Nothing, ScanBinner}
    # The scantool's configuration and state on the last train, the followed
    # motor's index into its active motors (nothing if none) and its alias.
    motor_config::Any
    state::String
    motor_idx::Union{Nothing, Int}
    motor_alias::String
    # Label of the running scan, nothing between scans
    scan_label::Union{Nothing, String}
    records::OrderedDict{String, VariableData}
end

function Scan(; automatic=true, scantool=nothing, motor=nothing, position_property="actualPosition",
              resolution=1e-3, n_scans=10, history_budget=64 * 1024 * 1024, motor_priority=String[])
    Scan(Parameter{Bool}(; value=automatic, update_handler=scan_rewire),
         Parameter{KaraboDevice}(; value=scantool, update_handler=scan_rewire, filter_device_classes=["Karabacon"]),
         Parameter{Dependency}(; value=motor, update_handler=scan_motor_changed, optional=true),
         Parameter(position_property), Parameter{Float64}(; value=resolution, update_handler=scan_rebin),
         Parameter(n_scans), Parameter(history_budget), motor_priority,
         nothing, nothing, "", nothing, "", nothing, OrderedDict{String, VariableData}())
end

default_name(::Scan) = "scan"

# A new mode or scantool changes the dependencies, which are wired on restart
scan_rewire(::Scan, _) = request_rewire(current_ctx)

# The bins were over another motor
function scan_motor_changed(s::Scan, _)
    s.binner = nothing
    request_rewire(current_ctx)
end

function scan_rebin(s::Scan, _)
    if !isnothing(s.binner)
        rebin!(s.binner)
    end
end

function device_classes(s::Scan)
    if s.automatic[]
        return Dict{Symbol, Any}(:scantool => "Karabacon")
    else
        return Dict{Symbol, Any}()
    end
end

# The scantool's state and configuration (see `follow_scantool!`) in automatic
# mode, then the motor's position once it's known.
function postprocessor_dependencies(s::Scan)
    deps = Dependency[]
    if s.automatic[] && isassigned(s.scantool)
        dev = s.scantool[]
        topic = isempty(dev.topic) ? nothing : dev.topic
        for property in ("state", "scanEnv.scanType", "scanEnv.startPoints", "scanEnv.stopPoints",
                         "scanEnv.steps", "deviceEnv.activeMotorDeviceIds", "deviceEnv.activeMotors")
            push!(deps, karabo_dependency(topic, dev.name, property))
        end
    end
    if isassigned(s.motor)
        push!(deps, s.motor[])
    end
    return deps
end

get_record(s::Scan, key) = s.records[key]

# The call matches `postprocessor_dependencies`: no motor yet, a motor alone, or
# the scantool's properties followed by the motor once it's known.
(s::Scan)(data) = scan_output(s)

# A motor in automatic mode is only binned along with its scantool
function (s::Scan)(data, position)
    if !s.automatic[]
        bin_scan!(s, data, position)
    end
    return scan_output(s)
end

function (s::Scan)(data, state, scan_type, starts, stops, steps, motor_ids, aliases, position=nothing)
    follow_scantool!(s, state, scan_type, starts, stops, steps, motor_ids, aliases)
    # Only the running scan is binned, along its motor
    if !isnothing(s.scan_label) && !isnothing(s.motor_idx)
        bin_scan!(s, data, position)
    end
    return scan_output(s)
end

function bin_scan!(s::Scan, data, position)
    if !isnothing(position) && !isnothing(data)
        if isnothing(s.binner)
            s.binner = ScanBinner(1; names=(:position, :position2))
        end
        push!(s.binner, data, position; resolutions=[s.resolution[]], history_budget=s.history_budget[])
    end
end

function scan_output(s::Scan)
    bins = scan_bins(s)
    records = collect(keys(s.records))
    return isnothing(bins) ? VariableData(; data=nothing, records) : (@set bins.records = records)
end

# Follow the scantool's configuration and scans. The 1-D scan types move all
# active motors along a single axis, so one of them is followed (see
# `motor_priority`); mesh scans aren't supported yet. A scan starts when the
# state enters ACQUIRING other than from PAUSED (a resume) and ends when it's
# back to ON, or ERROR for an aborted scan. Step scans set the resolution from
# their step size, other scans have no useful one.
function follow_scantool!(s::Scan, state, scan_type, starts, stops, steps, motor_ids, aliases)
    # The trainmatcher has no value for a property until it's first updated
    if !any(isnothing, (state, scan_type, starts, stops, steps, motor_ids, aliases))
        motors = String.(motor_ids)
        if (scan_type, motors) != s.motor_config
            s.motor_config = (scan_type, motors)
            s.motor_idx = if scan_type in ("mesh", "dmesh")
                @warn "Scan '$(Meta.name[])': mesh scans aren't supported yet" maxlog=1
                nothing
            elseif scan_type == "tscan" || isempty(motors)
                nothing
            else
                preferred = findfirst(in(motors), s.motor_priority)
                isnothing(preferred) ? 1 : findfirst(==(s.motor_priority[preferred]), motors)
            end

            # The aliases are listed in the same order as the device ids
            if isnothing(s.motor_idx)
                s.motor_alias = ""
            else
                s.motor_alias = String(aliases[s.motor_idx])
                dev = s.scantool[]
                motor = karabo_dependency(isempty(dev.topic) ? nothing : dev.topic, motors[s.motor_idx],
                                          s.position_property[])
                if !isassigned(s.motor) || s.motor[] != motor
                    tryset(s.motor, motor; force=true)
                    @info "Scan '$(Meta.name[])' follows $(motor.source) of scantool '$(dev.name)'"
                    request_rewire(current_ctx)
                end
            end
        end

        state = String(state)
        if state == "ACQUIRING" && s.state ∉ ("ACQUIRING", "PAUSED")
            s.scan_label = "$(Libc.strftime("%Y-%m-%d %H:%M:%S", time())) ($(join(aliases, ", ")))"
            s.binner = nothing
            # Half a step snaps positions onto the nearest level. Start/stop
            # points are per motor (a single one applies to every motor), the
            # step count is shared.
            if scan_type in ("ascan", "dscan") && !isnothing(s.motor_idx)
                i = s.motor_idx
                step = abs(Float64(stops[min(i, end)] - starts[min(i, end)]))
                resolution = step / max(steps[1], 1) / 2
                if resolution > 0
                    tryset(s.resolution, resolution)
                end
            end
        elseif state in ("ON", "ERROR") && !isnothing(s.scan_label)
            bins = scan_bins(s)
            if !isnothing(bins)
                s.records[s.scan_label] = bins
                while length(s.records) > s.n_scans[]
                    delete!(s.records, first(keys(s.records)))
                end
            end
            s.scan_label = nothing
        end
        s.state = state
    end
end

# The current bins, or nothing before anything was binned
function scan_bins(s::Scan)
    if isnothing(s.binner) || isnothing(s.binner.seq)
        return nothing
    else
        motor = Symbol(if !isempty(s.motor_alias)
            s.motor_alias
        elseif isassigned(s.motor)
            s.motor[].name
        else
            "position"
        end)
        # The axes are labelled by the dims: the position after the motor, the
        # values and an anonymous value dim after the binned output. They're
        # rebuilt since DD.set can't handle the binner's plain vector lookups.
        source = Symbol(first(rsplit(Meta.name[], '.'; limit=2)))
        seq = s.binner.seq
        dims = map(DD.dims(seq.mean)) do dim
            if DD.name(dim) == :position
                DD.Dim{motor}(DD.val(dim))
            elseif DD.name(dim) == :dim_1
                DD.Dim{source}(DD.val(dim))
            else
                dim
            end
        end
        data = DD.rebuild(seq.mean; dims, name=source)
        # The std is pooled over a vector's elements, so only scalar scans get one
        uncertainty = ndims(seq.mean) == 1 ? DD.rebuild(seq.std; dims) : nothing
        return VariableData(; data, uncertainty, compress=false, fixed_aspect=false)
    end
end
