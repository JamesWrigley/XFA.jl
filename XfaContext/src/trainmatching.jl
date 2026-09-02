# Stand-in payload sent in place of an unsubscribed array. Carries just enough
# shape info for the client to display plot buttons / type labels without
# needing the full data.
struct ArrayMetadata
    eltype::DataType
    size::Vector{Int}
end

# Exponentially-weighted running estimate of an event rate (Hz). Call
# `tick!(rr)` on each event; `value` is NaN until the second tick (a single
# event has no inter-arrival interval). The first interval seeds the EMA
# directly so it doesn't have to ramp up from zero.
mutable struct RunningRate
    α::Float64
    last_ts::Float64
    value::Float64
end
RunningRate(α::Real=0.2) = RunningRate(α, NaN, NaN)

function tick!(rr::RunningRate)
    now_ts = time()
    if !isnan(rr.last_ts) && now_ts > rr.last_ts
        instantaneous = 1 / (now_ts - rr.last_ts)
        rr.value = isnan(rr.value) ? instantaneous :
            rr.α * instantaneous + (1 - rr.α) * rr.value
    end
    rr.last_ts = now_ts
    return rr.value
end

# One mark within a PlotSpec. `data` names the variable to render (the client
# subscribes to it). `mark` picks the ImPlot primitive. The axis channels say
# where each axis's values come from: a String pulls from a sibling variable, a
# Symbol selects a named dimension of `data` (e.g. a DimArray dim), and nothing
# lets the client infer it from the data. A grouping channel bound to a dim
# (e.g. color=:foo) fans the layer out into one series per coordinate along that
# dim; set `gradient` to shade those series along a single-hue gradient instead
# of distinct colors (useful when there are many). Modelled on Vega-Lite's
# mark/encoding split so the same data can be plotted different ways by
# reshaping channels.
@kwdef struct LayerSpec
    data::String
    mark::Symbol = :line          # :line, :scatter, :bars, :image
    x::Union{String, Symbol, Nothing} = nothing
    y::Union{String, Symbol, Nothing} = nothing
    color::Union{String, Symbol, Nothing} = nothing
    gradient::Bool = false
    label::Union{String, Nothing} = nothing
end

# A named, openable plot a variable advertises in addition to its default
# output. The client opens it by `name`, which also keys reconciliation across
# trains: name/title/labels/layers may change per train and the client reshapes
# an already-open plot to match. Layers are stacked into one plot.
@kwdef struct PlotSpec
    name::String
    layers::Vector{LayerSpec}
    title::Union{String, Nothing} = nothing
    xlabel::Union{String, Nothing} = nothing
    ylabel::Union{String, Nothing} = nothing
end

PlotSpec(name::AbstractString, layers::AbstractVector{LayerSpec}; kwargs...) =
    PlotSpec(; name=String(name), layers=collect(layers), kwargs...)

# Sugar: a vector of variable names becomes one default line layer per name.
PlotSpec(name::AbstractString, vars::AbstractVector{<:AbstractString}; kwargs...) =
    PlotSpec(name, [LayerSpec(; data=String(v)) for v in vars]; kwargs...)

@kwdef struct VariableData{T}
    tid::Int = 0
    name::Union{String, Nothing} = nothing
    data::T
    subvariables::Dict{String, Any} = Dict{String, Any}()
    title::Union{String, Nothing} = nothing
    x_axis::Union{AbstractVector, Nothing} = nothing
    y_axis::Union{AbstractVector, Nothing} = nothing
    xlabel::Union{String, Nothing} = nothing
    ylabel::Union{String, Nothing} = nothing
    unit::Union{String, Nothing} = nothing
    bin_resolution::Float64 = 0.0
    fixed_aspect::Bool = true
    plot_type::Symbol = :series
    plot_specs::Vector{PlotSpec} = PlotSpec[]
    update_rate::Float64 = 0.0
    compress::Bool = true
end

VariableData(tid, name, data) = VariableData(; tid=Int(tid), name, data)
VariableData(tid, name, data, subvariables) = VariableData(; tid=Int(tid), name, data, subvariables)

# `update_rate` is a runtime metric, not part of value identity, so it's
# excluded from equality and hashing.
function Base.:(==)(x::VariableData{T}, y::VariableData{T}) where {T}
    for f in fieldnames(VariableData)
        if f === :update_rate
            continue
        end
        if getfield(x, f) != getfield(y, f)
            return false
        end
    end
    return true
end

function Base.hash(x::VariableData, h::UInt)
    for f in fieldnames(VariableData)
        if f === :update_rate
            continue
        end
        v = getfield(x, f)
        if f === :subvariables && isempty(v)
            h = hash(0, h)
        else
            h = hash(v, h)
        end
    end
    return h
end

mutable struct Trainmatcher
    max_train_latency::Int
    sources::Set{String}
    train_data::Dict{Int}
    latest_trainid::Int

    """
        Trainmatcher(sources, max_train_latency::Int)

    Create a Trainmatcher object, which tries to match `sources` coming from a
    Karabo bridge. `sources` is some iterable of `String`'s. The matching is
    'greedy', which means that if not all sources have been received for a certain
    train after `max_train_latency` trains, then the incomplete train data will be
    dropped. A negative `max_train_latency` disables this: incomplete trains are
    kept indefinitely (used for lossless offline replay).
    """
    function Trainmatcher(sources, max_train_latency::Integer=20)
        new(max_train_latency, Set(sources), Dict{Int, Any}(), -1)
    end
end

"""
    match_train!(matched_trains, tm::Trainmatcher, variable::VariableData)

Match `variable` with the trains already in `tm` and write the matched trains to
`matched_trains`.
"""
function match_train!(matched_trains::Dict{Int, Any}, tm::Trainmatcher, variable::VariableData)
    if variable.name ∉ tm.sources
        throw(ArgumentError("Variable '$(variable.name)' is not in the list of sources to match"))
    end

    # Update cached data
    tm.latest_trainid = max(tm.latest_trainid, variable.tid)
    if !haskey(tm.train_data, variable.tid)
        tm.train_data[variable.tid] = Dict{String, Any}()
    end
    tm.train_data[variable.tid][variable.name] = variable

    # Pop trains that are too old, or fully matched
    for tid in collect(keys(tm.train_data))
        if issetequal(tm.sources, keys(tm.train_data[tid]))
            matched_trains[tid] = pop!(tm.train_data, tid)
        elseif tm.max_train_latency >= 0 && tm.latest_trainid - tid > tm.max_train_latency
            pop!(tm.train_data, tid)
        end
    end

    return matched_trains
end

"""
    match_train(tm::Trainmatcher, variable::VariableData)

Non-modifying version of `match_train!()`.
"""
function match_train(tm::Trainmatcher, variable::VariableData)
    matched_trains = Dict{Int, Any}()
    match_train!(matched_trains, tm, variable)
end
