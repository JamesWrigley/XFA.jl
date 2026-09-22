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

# Plot specs are the typed form of a Vega-Lite spec, restricted to the subset
# the client draws: a PlotSpec is a layered spec and a LayerSpec one of its unit
# specs, with the same vocabulary (mark, encoding channels, lookup transform,
# params).

@enum Mark Mark_Line Mark_Point Mark_Bar Mark_Rect
@enum FieldType FieldType_Quantitative FieldType_Nominal FieldType_Ordinal
@enum LookupKey LookupKey_TrainId LookupKey_Index
@enum ModelFunction ModelFunction_Gaussian

# An encoding channel. `field` is a field of the layer's variable: a dim by name
# or position (`index`, or `row`/`col`), `value`, or `trainId` for a scalar
# history. On a matrix grouped into series by a color channel, `index` runs
# along each series. `title` is "" when hidden, and nothing when left to the
# data's own labels.
struct ChannelDef
    field::String
    type::FieldType
    title::Union{String, Nothing}
    log::Bool
    scheme::Union{String, Nothing}
    binned::Bool
end

# Pulls `field` of `dataset` into the layer as `as`, matched on `key`.
struct LookupTransform
    key::LookupKey
    dataset::String
    field::String
    as::String
end

struct LayerSpec
    data::String
    mark::Mark
    opacity::Float64
    x::ChannelDef
    y::ChannelDef
    color::Union{ChannelDef, Nothing}
    lookup::Union{LookupTransform, Nothing}
end

# An interval selection named after the parameter it edits. `initial` is
# unassigned when the spec gives no value.
struct RoiParam
    name::String
    initial::AbstractROI
end

# A model curve drawn over the spec's layers, its parameters read from dataset
# `params`.
struct ModelOverlay
    func::ModelFunction
    params::String
    title::Union{String, Nothing}
end

# Sugar: the model function by name (:gaussian).
function ModelOverlay(func::Symbol; params::AbstractString,
                      title::Union{AbstractString, Nothing}=nothing)
    functions = (gaussian=ModelFunction_Gaussian,)
    if !haskey(functions, func)
        throw(ArgumentError("unsupported model function :$(func), expected one of: $(join(keys(functions), ", "))"))
    end
    ModelOverlay(functions[func], String(params), isnothing(title) ? nothing : String(title))
end

# A named plot a variable advertises on `VariableData.plot_specs`, besides its
# default one. The user opens it by `name`; the rest may change from train to
# train and an open plot follows it.
struct PlotSpec
    name::String
    title::String
    xlabel::Union{String, Nothing}
    ylabel::Union{String, Nothing}
    layers::Vector{LayerSpec}
    rois::Vector{RoiParam}
    models::Vector{ModelOverlay}
    fixed_aspect::Bool
end

function Base.:(==)(a::PlotSpec, b::PlotSpec)
    all(f -> getfield(a, f) == getfield(b, f), fieldnames(PlotSpec))
end

function Base.hash(spec::PlotSpec, h::UInt)
    for f in fieldnames(PlotSpec)
        h = hash(getfield(spec, f), h)
    end
    return h
end

# Layers share their axes: unless given, each label is the first layer's
# channel title that isn't hidden.
function PlotSpec(name::AbstractString, layers::AbstractVector{LayerSpec}=[LayerSpec()];
                  title::AbstractString=name,
                  xlabel::Union{AbstractString, Nothing}=nothing,
                  ylabel::Union{AbstractString, Nothing}=nothing,
                  rois::Vector{RoiParam}=RoiParam[], models::Vector{ModelOverlay}=ModelOverlay[],
                  fixed_aspect::Bool=true)
    x_title = y_title = ""
    for layer in layers
        if x_title == ""
            x_title = layer.x.title
        end
        if y_title == ""
            y_title = layer.y.title
        end
    end
    PlotSpec(String(name), String(title), isnothing(xlabel) ? x_title : xlabel, isnothing(ylabel) ? y_title : ylabel,
             collect(layers), rois, models, fixed_aspect)
end

# `spec` with the layers that don't name a variable ("") pointed at `variable`,
# the one advertising it.
function bind_variable(spec::PlotSpec, variable)
    if all(layer -> !isempty(layer.data), spec.layers)
        spec
    else
        @set spec.layers = [isempty(layer.data) ? (@set layer.data = variable) : layer for layer in spec.layers]
    end
end

# Lets `VariableData(; plot_specs = spec)` take a lone spec.
Base.convert(::Type{Vector{PlotSpec}}, spec::PlotSpec) = [spec]

# Sugar: a vector of variable names becomes one default line layer per name.
PlotSpec(name::AbstractString, vars::AbstractVector{<:AbstractString}; kwargs...) =
    PlotSpec(name, [LayerSpec(; data=v) for v in vars]; kwargs...)

# Sugar for a layer drawing variable `data` (by default the one advertising the
# spec, see bind_variable) with `mark` (:line, :scatter, :bars or :image). `x`
# is a dim of `data` (a Symbol), or for 1D data another variable
# to plot against elementwise (a String); by default the index. `y` is only for
# images, which take the dim along each axis. `color` is a dim to group a matrix
# by, into one series per coordinate (with `x` along the other dim by default),
# shaded along a continuous scheme unless `gradient` is false, which gives them
# distinct colours.
function LayerSpec(; data::AbstractString="", mark::Symbol=:line,
                   x::Union{AbstractString, Symbol, Nothing}=nothing, y::Union{Symbol, Nothing}=nothing,
                   color::Union{Symbol, Nothing}=nothing, gradient::Bool=true)
    marks = (line=Mark_Line, scatter=Mark_Point, bars=Mark_Bar, image=Mark_Rect)
    if !haskey(marks, mark)
        throw(ArgumentError("unsupported mark :$(mark), expected one of: $(join(keys(marks), ", "))"))
    end
    # The axes are labelled by the data
    axis(field) = ChannelDef(String(field), FieldType_Quantitative, nothing, false, nothing, false)
    colored(field, type, scheme) = ChannelDef(String(field), type, String(field), false, scheme, false)

    if mark == :image
        if x isa AbstractString || !isnothing(color)
            throw(ArgumentError("an image takes a dim for x and y, and no color"))
        end
        LayerSpec(String(data), Mark_Rect, 1.0, axis(something(x, "col")), axis(something(y, "row")),
                  colored("value", FieldType_Quantitative, "turbo"), nothing)
    else
        if !isnothing(y)
            throw(ArgumentError("y is only supported for images"))
        elseif !isnothing(color) && x isa AbstractString
            throw(ArgumentError("color groups a matrix by a dim, so x must be its other dim, not a variable"))
        end
        color_channel = if isnothing(color)
            nothing
        elseif gradient
            colored(color, FieldType_Quantitative, "viridis")
        else
            colored(color, FieldType_Nominal, nothing)
        end
        lookup = x isa AbstractString ? LookupTransform(LookupKey_Index, String(x), "value", String(x)) : nothing
        LayerSpec(String(data), marks[mark], 1.0, axis(something(x, "index")), axis("value"),
                  color_channel, lookup)
    end
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
    bin_resolution::Float64 = 0.0
    fixed_aspect::Bool = true
    plot_type::Symbol = :series
    plot_specs::Vector{PlotSpec} = PlotSpec[]
    update_rate::Float64 = 0.0
    compress::Bool = true
end

VariableData(data; kwargs...) = VariableData(; data, kwargs...)
VariableData(tid, name, data) = VariableData(; tid=Int(tid), name, data)
VariableData(tid, name, data, subvariables) = VariableData(; tid=Int(tid), name, data, subvariables)

# The default constructor of a parametric struct doesn't convert its arguments,
# this one does (e.g. a lone PlotSpec for `plot_specs`).
VariableData(tid, name, data::T, fields...) where {T} = VariableData{T}(tid, name, data, fields...)

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
    match_train!(matched_trains, dropped_trains, tm::Trainmatcher, variable::VariableData)

Match `variable` with the trains already in `tm` and write the matched trains to
`matched_trains`. Trains given up on because they stayed incomplete for longer
than `max_train_latency` are appended to `dropped_trains`. Data for a dropped
train arriving later is matched afresh (slow devices legitimately send old
train IDs), so such a train can be dropped more than once.
"""
function match_train!(matched_trains::Dict{Int, Any}, dropped_trains::Vector{Int},
                      tm::Trainmatcher, variable::VariableData)
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
            push!(dropped_trains, tid)
        end
    end

    return matched_trains
end

"""
    match_train(tm::Trainmatcher, variable::VariableData)

Non-modifying version of `match_train!()`, returning only the matched trains.
"""
function match_train(tm::Trainmatcher, variable::VariableData)
    matched_trains = Dict{Int, Any}()
    match_train!(matched_trains, Int[], tm, variable)
end
