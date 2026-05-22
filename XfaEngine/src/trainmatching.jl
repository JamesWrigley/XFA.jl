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
    fixed_aspect::Bool = true
    plot_type::Symbol = :series
    update_rate::Float64 = 0.0
end

VariableData(tid, name, data) = VariableData(; tid=Int(tid), name, data)
VariableData(tid, name, data, subvariables) = VariableData(; tid=Int(tid), name, data, subvariables)

# `update_rate` is a runtime metric, not part of value identity, so it's
# excluded from equality and hashing.
function Base.:(==)(x::VariableData{T}, y::VariableData{T}) where {T}
    (x.tid == y.tid && x.name == y.name && x.data == y.data && x.subvariables == y.subvariables
     && x.title == y.title && x.x_axis == y.x_axis && x.y_axis == y.y_axis
     && x.xlabel == y.xlabel && x.ylabel == y.ylabel && x.unit == y.unit
     && x.fixed_aspect == y.fixed_aspect && x.plot_type == y.plot_type)
end

function Base.hash(x::VariableData, h::UInt)
    subvariable_hash = isempty(x.subvariables) ? hash(0) : hash(x.subvariables)
    hash(x.tid, hash(x.name, hash(x.data, hash(subvariable_hash,
         hash(x.title, hash(x.x_axis, hash(x.y_axis, hash(x.xlabel, hash(x.ylabel,
              hash(x.unit, hash(x.fixed_aspect, hash(x.plot_type, h))))))))))))
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
    returned.
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
        elseif tm.latest_trainid - tid > tm.max_train_latency
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
