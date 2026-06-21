module XfaContext

export @karabo_str, @Variable, @Input, @Group, @add_subvariable, @display, Parameter, tryset, KaraboDevice,
    Dependency, DependencyKind, DepKind_Variable, DepKind_Subvariable, DepKind_Karabo, DepKind_Group, DepKind_GroupParameter,
    karabo_dependency, subvariable_dependency, group_dependency, group_parameter_dependency,
    RectROI, Context

# Macro-generated code and context files refer to this module as `Context`. It's
# exported so `using XfaContext` binds it the way `using XfaEngine.Context` used
# to, and it doubles as a back-compat alias for the engine.
const Context = @__MODULE__

# Capacity of the per-train variable channels (see variable_channel). Also used
# as the depth of the engine's Karabo recv buffer pool, so a recycled buffer has
# had a full window of trains to drain through every consumer.
const VARIABLE_CHANNEL_SIZE = 100

import Base.ScopedValues: @with

using DistributedNext: DistributedNext, RemoteChannel, remote_do, remoteref_id, remotecall_fetch

import MacroTools
import MacroTools: @capture, postwalk, prettify
import OrderedCollections: OrderedDict
using Accessors: @set
using DimensionalData: DimensionalData, DimensionalData as DD

using DataStructures: DataStructures, CircularBuffer, isfull
include("circular_channel.jl")

# Factory for channels carrying per-train variable data. Online (`offline=false`)
# returns a RemoteChannel over a CircularChannel: oldest items are overwritten
# when a consumer falls behind, and drops are counted per channel. Offline
# returns a plain bounded Channel, whose `put!` blocks instead of dropping so a
# finite run is replayed losslessly.
function variable_channel(offline::Bool=false)
    if offline
        Channel{VariableData}(VARIABLE_CHANNEL_SIZE)
    else
        RemoteChannel(() -> CircularChannel{VariableData}(VARIABLE_CHANNEL_SIZE))
    end
end

struct ChannelStat
    drops::Int
    size::Int
    capacity::Int
end

# Snapshot of (drops, current size, capacity) for a RemoteChannel-wrapped
# CircularChannel. Runs on the worker that owns the channel.
function channel_stat(rc::RemoteChannel)::ChannelStat
    remotecall_fetch(rc.where, rc) do rc
        ch = DistributedNext.channel_from_id(remoteref_id(rc))
        ChannelStat(drop_count(ch), size(ch), DataStructures.capacity(ch))
    end
end

# Trait functions for dispatch-based metadata registration.
# Overloaded by @Variable, @Input, and @Group macros for each
# function/type they define, enabling Revise compatibility.
function variable_dependencies end
function input_dependencies end
function group_fields end
variable_subvariables(_) = String[]
variable_postprocessors(_) = VariablePostprocessor[]
variable_displays(_) = String[]

# Returns the topic served by an input group, or nothing for generic inputs
input_topic(::Any) = nothing

# Returns the identifying device for an input group (used to match routing
# rules against the input), or nothing if the input isn't device-backed.
input_device(::Any) = nothing

# Returns the sources available from an input group
get_sources(::Any) = String[]
# For variable references (@Variable name -> MyLib.func), returns the
# original function. Used to exclude origin functions from the context
# when they are already represented by a reference wrapper.
variable_origin(f) = f

struct Neighbour
    name::String
    channel::Union{RemoteChannel, Channel}
end

# Container module for train/variable-specific information. This conflicts with
# Base.Meta but we accept that for the convenience of the name. In this module
# and the context file, usage of Base.Meta should always explicitly refer to
# `Base.Meta` to avoid confusion.
module Meta

import Base.ScopedValues: ScopedValue

const tid = ScopedValue{Int}()
const run_number = ScopedValue{Int}()
const proposal = ScopedValue{Int}()
const name = ScopedValue{String}()

const scratch = ScopedValue(Dict{String, Any}())
const subvariables = ScopedValue(Dict{String, Any}())

end

include("context_types.jl")
include("trainmatching.jl")
include("detector_assembly.jl")
include("binning.jl")

using DataStructures: CircularBuffer
using FHist: FHist, Hist1D, Hist2D, bincounts, bincenters, binedges
using NaNStatistics: NaNStatistics, nanmean, nansum, nanmean!, nansum!, allocate_nanmean, allocate_nansum
include("context_builtins.jl")

@kwdef mutable struct WorkerState
    task_locks = Dict{String, ReentrantLock}()
    dag_functions = Dict{String, Function}()
    parameters = Dict{String, Parameter}()
    current_ctx_module::Module = Module()
end

@kwdef mutable struct ContextState
    functions::Dict{String, Any} = Dict()
    group_types::Dict{DataType, Group} = Dict()
    groups::Dict{String, Any} = Dict()
    dag::Dict{String, OrderedDict} = Dict()
    subvariables::Dict{String, Vector{String}} = Dict()
    variable_postprocessors::Dict{String, Vector{String}} = Dict()
    # Variable name -> list of fully-qualified parameter names to overlay
    # on the variable's plot. Populated from @display declarations.
    displays::Dict{String, Vector{String}} = Dict()
    postprocessors::Dict{String, AbstractPostprocessor} = Dict()
    parameters::Dict{String, Parameter} = Dict()
    exprs::Vector{Expr} = Expr[]

    inputs::Dict{String, Any} = Dict()
    prelude::Vector{Expr} = Expr[]
    input_channels::Dict{String, Channel} = Dict()
    input_tasks::Dict{String, Task} = Dict()
    available_sources::Dict{String, Vector{String}} = Dict()

    dep_to_input::Dict{String, String} = Dict()

    input_variable_channels::Dict{String, Dict{String, Union{RemoteChannel, Channel}}} = Dict()
    input_variables_tasks::Dict{String, Task} = Dict()

    external_dependency_channels::Dict{String, Dict{String, Union{RemoteChannel, Channel}}} = Dict()
    external_dependency_tasks::Dict{String, Task} = Dict()

    variable_tasks::Dict{String, Task} = Dict()
    variable_channels::Dict{String, Dict{String, Union{RemoteChannel, Channel}}} = Dict()

    # Smoothed Hz at which inputs and external dependencies are pushing data,
    # keyed by input/dep name. Variable rates are sent piggy-backed on
    # `VariableData.update_rate` instead.
    input_rates::Dict{String, Float64} = Dict()

    stream_output::Union{RemoteChannel, Channel, Nothing} = nothing
    forwarder::Function = Returns(nothing)
    output_forwarder_task::Union{Task, Nothing} = nothing

    # Invoked on proc 1 when a parameter changes value via `tryset`, as
    # (name, value). The engine installs a closure that broadcasts the change
    # to connected clients; offline runners leave it as the no-op default.
    on_parameter_changed::Function = Returns(nothing)

    path::String = ""

    # Offline-replay plan, populated on a copy of a loaded context by `run`. The
    # feeder replaces all of the context's inputs (it pushes `(tid, data)` like a
    # real @Input). `variable_overrides` maps a variable name to a value emitted
    # per train in place of computing it; an overridden variable has its
    # dependencies cut (so its upstream is pruned unless separately requested)
    # and is driven as a root by `matched_tids`, the run's train list.
    input_feeder::Union{Function, Nothing} = nothing
    variable_overrides::Dict{String, Any} = Dict()
    matched_tids::Vector{Int} = Int[]

    is_running::Threads.Atomic{Bool} = Threads.Atomic{Bool}(false)
    events_channel::Union{RemoteChannel, Nothing} = nothing
    watcher_task::Task = Task(Returns(nothing))
end

worker_state::WorkerState = WorkerState()
current_ctx::Union{ContextState, Nothing} = nothing

function Base.show(io::IO, ctx::ContextState)
    n_variables = length(ctx.functions)
    n_params = length(ctx.parameters)
    print(io, "ContextState($(n_variables) variables, $(n_params) parameters)")
end

"""
Finds all external dependencies (i.e. from Karabo) required by the context.
"""
function external_dependencies(ctx::ContextState; per_variable=false)
    deps_per_variable = Dict{String, Vector{Dependency}}()
    all_deps = Dependency[]

    for (name, deps) in ctx.dag
        for (_, dep) in deps
            if dep isa Dependency && dep.kind == DepKind_Karabo
                deps_vec = get!(deps_per_variable, name, Dependency[])
                push!(deps_vec, dep)
                push!(all_deps, dep)
            end
        end
    end

    return per_variable ? deps_per_variable : unique(all_deps)
end

# Returns the group object for an input, or nothing if it has no group.
function get_input_group(ctx::ContextState, input_name)
    for (_, dep) in ctx.inputs[input_name]
        if dep isa Dependency && dep.kind == DepKind_Group
            return ctx.groups[dep.name]
        end
    end
    return nothing
end

# Build a mapping from external dependency name -> input name.
# Routing priority:
# 1. Routing rule: match (topic, source) against engine-level rules; the
#    resolved trainmatcher device name is looked up against the inputs.
# 2. Topic match: dep has a topic, input group's input_topic() matches
# 3. Source match: dep's source is in the input group's get_sources() result
# 4. Single input fallback: only one input exists
function build_dep_routing(ctx::ContextState, dep_router=Returns(nothing))
    dep_to_input = Dict{String, String}()
    deps = external_dependencies(ctx)

    if isempty(ctx.inputs) || isempty(deps)
        return dep_to_input
    end

    # Build topic -> input, sources -> input, and trainmatcher-device -> input maps
    topic_map = Dict{String, String}()
    source_map = Dict{String, String}()
    device_map = Dict{KaraboDevice, String}()
    for (input_name, _) in ctx.inputs
        group = get_input_group(ctx, input_name)
        if isnothing(group)
            continue
        end

        topic = input_topic(group)
        if !isnothing(topic)
            topic_map[topic] = input_name
        end

        device = input_device(group)
        if !isnothing(device)
            device_map[device] = input_name
        end

        for source in get_sources(group)
            source_map[source] = input_name
        end
    end

    for dep in deps
        dep_name = string(dep)

        # 1. Routing. `dep_router` (supplied by the engine) maps (topic, source)
        # to an input device string, which we parse as a KaraboDevice —
        # topic-qualified ("T//DEV") matches exactly, bare ("DEV") matches by
        # name only (first-hit wins if multiple topics share a device name).
        dep_topic = isnothing(dep.topic) ? "" : dep.topic
        matched = dep_router(dep_topic, dep.source)
        if !isnothing(matched)
            target = KaraboDevice(matched)
            input_name = if !isempty(target.topic)
                get(device_map, target, nothing)
            else
                name_hits = [v for (dev, v) in device_map if dev.name == target.name]
                isempty(name_hits) ? nothing : first(name_hits)
            end
            if !isnothing(input_name)
                dep_to_input[dep_name] = input_name
                continue
            end
        end

        # 2. Topic match
        if !isnothing(dep.topic) && haskey(topic_map, dep.topic)
            dep_to_input[dep_name] = topic_map[dep.topic]
            continue
        end

        # 3. Source match
        if haskey(source_map, dep.source)
            dep_to_input[dep_name] = source_map[dep.source]
            continue
        end

        # 4. Single input fallback
        if length(ctx.inputs) == 1
            dep_to_input[dep_name] = first(keys(ctx.inputs))
            continue
        end

        throw(XfaContextException("Cannot determine which input serves dependency '$(dep_name)'. " *
                                  "Use a topic prefix (e.g. karabo\"TOPIC//$(dep_name)\") or ensure " *
                                  "the source is available from one of the inputs."))
    end

    return dep_to_input
end

# Returns the name used to match a dependency against a variable name.
# For most dependencies this is the full string, but for subvariable
# dependencies it's the parent name since subvariables share their parent's
# channel.
dep_variable_name(dep::Dependency) = dep.kind == DepKind_Subvariable ? dep.parent : dep.name

function find_downstream_neighbours(ctx::ContextState, dep_name, kind::DependencyKind)
    neighbours = Set{String}()
    for (var_name, deps) in ctx.dag
        for (_, dep) in deps
            if dep isa Dependency && dep.kind == kind && dep_variable_name(dep) == dep_name
                push!(neighbours, var_name)
            end
        end
    end

    return neighbours
end

# Return the origin path for a variable function or group type, stripping the
# anonymous context module prefix so that context-local definitions are just
# their name (e.g. "foo") and imported ones keep their module path
# (e.g. "MyLib.bar").
function origin_path(x)
    origin = x isa Function ? variable_origin(x) : x
    mod = parentmodule(origin)
    parts = String[]
    while !startswith(string(nameof(mod)), "UserContext")
        pushfirst!(parts, string(nameof(mod)))
        if mod === parentmodule(mod)
            break
        end
        mod = parentmodule(mod)
    end
    push!(parts, string(nameof(origin)))
    return join(parts, ".")
end

function to_dict(ctx::ContextState)
    inputs = Dict{String, Vector{String}}()
    for (name, deps) in ctx.inputs
        # Anonymous args (e.g. `::MockInput`) have `nothing` keys; skip those
        # since they carry no user-visible name.
        inputs[name] = [k for k in keys(deps) if !isnothing(k)]
    end

    groups = sort(collect(keys(ctx.groups)))

    origins = Dict{String, String}()
    for (name, func) in ctx.functions
        origins[name] = origin_path(func)
    end
    for (group_type, _) in ctx.group_types
        for (name, obj) in ctx.groups
            if obj isa group_type
                origins[name] = origin_path(group_type)
            end
        end
    end

    parameters = Dict{String, Parameter}()
    for (name, param) in ctx.parameters
        # Preserve the concrete type so an unassigned (nothing-valued) parameter
        # keeps its element type instead of failing to infer it.
        parameters[name] = typeof(param)(; name=param.name, value=param.value, set_by_user=param.set_by_user)
    end

    # group_type holds a DataType that may live in the context's anonymous
    # module, which doesn't exist on the client and would break deserialization.
    dag = Dict{String, OrderedDict}()
    for (name, deps) in ctx.dag
        # Unassigned optional dependencies aren't real edges, so they're left out
        # of the DAG. The client still learns about the parameter itself via the
        # `parameters` entry below, so it can be drawn in the GUI.
        dag[name] = OrderedDict{Any, Any}(k => v isa Dependency && !isnothing(v.group_type) ?
                                          (@set v.group_type = nothing) : v
                                          for (k, v) in deps
                                          if !(v isa Parameter && !isassigned(v)))
    end

    # For group variables, recover the original arg_name => group field mapping
    # from variable_dependencies. The DAG stores the resolved Dependency value,
    # losing the field name needed for source rewriting on the client.
    group_parameter_args = Dict{String, Dict{String, String}}()
    for (var_name, func) in ctx.functions
        # Only group-instance variables (e.g. "foo_group.foo") need the mapping;
        # the unbound function entry (e.g. "foo") shares the same dep methods
        # but isn't tied to a constructor the client could rewrite.
        if !occursin('.', var_name) || !@invokelatest(applicable(variable_dependencies, func))
            continue
        end

        mapping = Dict{String, String}()
        for (arg_name, dep) in @invokelatest variable_dependencies(func)
            if dep isa Dependency && dep.kind == DepKind_GroupParameter
                mapping[arg_name] = dep.parameter
            end
        end
        if !isempty(mapping)
            group_parameter_args[var_name] = mapping
        end
    end

    postprocessor_origins = Dict{String, String}()
    for (pp_name, pp) in ctx.postprocessors
        postprocessor_origins[pp_name] = origin_path(typeof(pp))
    end

    return Dict("dag" => dag,
                "subvariables" => ctx.subvariables,
                "postprocessors" => ctx.variable_postprocessors,
                "postprocessor_origins" => postprocessor_origins,
                "displays" => ctx.displays,
                "parameters" => parameters,
                "inputs" => inputs,
                "groups" => groups,
                "origins" => origins,
                "dep_to_input" => ctx.dep_to_input,
                "group_parameter_args" => group_parameter_args,
                "path" => ctx.path)
end

"""
Return a sorted order of functions in the context to execute.

Subvariables are taken into account, but excluded from the output since only
their parents are functions that can be executed.
"""
function topological_sort(dag)
    # First we create a copy of the DAG with some changes:
    # - Remove all Parameter dependencies (TODO: don't do this when parameter
    #   functions are implemented)
    # - All external dependencies (i.e from Karabo) removed
    # - All subvariables represented by their parent function
    # - All dependencies converted to strings for simplicity
    # - All group object arguments are removed
    internal_dag = Dict{String, Vector{String}}()
    for (name, deps) in dag
        internal_dag[name] = [x isa Dependency && x.kind == DepKind_Subvariable ? x.parent : string(x)
                              for x in values(deps)
                              if !(x isa Dependency && x.kind ∈ (DepKind_Karabo, DepKind_Group)) && !(x isa Parameter)]
    end

    sorted_graph = String[]
    working_set = Set([name for (name, deps) in internal_dag if isempty(deps)])

    while !isempty(working_set)
        variable = pop!(working_set)
        push!(sorted_graph, variable)

        for (name, deps) in internal_dag
            idxs = findall(x -> x == variable, deps)

            if !isempty(idxs)
                deleteat!(deps, idxs)

                if isempty(deps)
                    push!(working_set, name)
                end
            end
        end
    end

    if !all(isempty.(values(internal_dag)))
        throw(XfaContextException("Context graph has a cycle, cannot construct a DAG"))
    end

    return sorted_graph
end

topological_sort(ctx::ContextState) = topological_sort(ctx.dag)

function execute_variables(ctx::ContextState, inputs::Dict)
    execution_order = topological_sort(ctx)
    results = Dict{String, Any}()

    for name in execution_order
        # Build up the argument list
        args = []
        for dep in values(ctx.dag[name])
            if dep isa Parameter
                push!(args, ctx.parameters[string(dep)].value)
            elseif dep.kind == DepKind_Karabo
                dep_key = dep.name
                if haskey(inputs, dep_key)
                    push!(args, inputs[dep_key])
                end
            elseif dep.kind == DepKind_Variable
                dep_key = dep.name
                if haskey(results, dep_key)
                    push!(args, results[dep_key])
                end
            elseif dep.kind == DepKind_Group
                push!(args, ctx.groups[dep.name])
            else
                throw(XfaContextException("Unrecognized dependency kind: $(dep.kind)"))
            end
        end

        # Call the function
        if length(args) == length(ctx.dag[name])
            try
                results[name] = @invokelatest ctx.functions[name](args...)
            catch ex
                @error "Error executing $(name)!" exception=ex
            end
        end
    end

    return results
end

function change_parameter(ctx::ContextState, new_param::Parameter)
    pause_pipeline() do
        ctx_param = worker_state.parameters[new_param.name]
        if !isnothing(ctx_param.update_handler)
            owner = find_parameter_owner(ctx, new_param.name)
            if !isnothing(owner)
                ctx_param.update_handler(owner, new_param.value)
            else
                ctx_param.update_handler(new_param.value)
            end
        end

        ctx_param.value = new_param.value
        ctx_param.set_by_user = true
    end
end

# Returns the owning group or postprocessor for a parameter, or nothing for
# top-level parameters. Owners are passed to update handlers so they can
# mutate additional state alongside the new value.
function find_parameter_owner(ctx::ContextState, param_name::String)
    dot_idx = findlast('.', param_name)
    isnothing(dot_idx) && return nothing
    prefix = param_name[1:dot_idx-1]

    haskey(ctx.postprocessors, prefix) && return ctx.postprocessors[prefix]
    haskey(ctx.groups, prefix) && return ctx.groups[prefix]

    return nothing
end

function input_wrapper(name, group, channel)
    f = worker_state.dag_functions[name]

    try
        if isnothing(group)
            @with Meta.name => name f(channel)
        else
            @with Meta.name => name f(group, channel)
        end
    catch ex
        if !(ex isa InvalidStateException)
            @error "Caught exception while executing input '$(name)'" exception=(ex, catch_backtrace())
        end
    finally
        close(channel)
    end
end

function wrap_result(result, tid, name; subvariables=Dict{String, Any}(), update_rate=0.0)
    if result isa VariableData
        v = @set result.tid = tid
        v = @set v.name = name
        v = @set v.subvariables = subvariables
        @set v.update_rate = update_rate
    elseif result isa Histogram1D
        VariableData(result; tid, name, subvariables, update_rate)
    else
        VariableData(; tid, name, data=result, subvariables, update_rate)
    end
end

function putall!(channels, value)
    for channel in channels
        put!(channel, value)
    end
end

function stream_input(name, channel, downstream_neighbours, rates)
    rate = RunningRate()
    try
        while isopen(channel) || isready(channel)
            tid, sources = take!(channel)

            tick!(rate)
            rates[name] = isnan(rate.value) ? 0.0 : rate.value

            putall!(values(downstream_neighbours), VariableData(; tid=Int(tid), data=sources))
            @debug "Pushed input data from '$(name)' to: $(keys(downstream_neighbours))"
        end
    catch ex
        if !(ex isa InvalidStateException)
            # If it's not an error about the channel being closed, show the exception
            @error "Couldn't get input data from '$(name)'" exception=(ex, catch_backtrace())
        end
    finally
        @debug "Finishing input '$(name)'"
        for neighbour_channel in values(downstream_neighbours)
            close(neighbour_channel)
        end
    end
end

function stream_external_dependency(name, input_neighbour, downstream_neighbours)
    channel = input_neighbour.channel

    dep = karabo_dependency(name)
    property_value = dep.property * ".value"

    try
        while isopen(channel) || isready(channel)
            input = take!(channel)
            data = nothing

            if haskey(input.data, dep.source)
                source_data = input.data[dep.source]
                if haskey(source_data, dep.property)
                    data = input.data[dep.source][dep.property]
                elseif haskey(source_data, property_value)
                    data = input.data[dep.source][property_value]
                end
            end

            result = VariableData(input.tid, name, data)
            putall!(values(downstream_neighbours), result)
            @debug "Pushed data for '$(name)' to: $(keys(downstream_neighbours))"
        end
    catch ex
        if !(ex isa InvalidStateException)
            @error "Executing external dependency '$(name)' failed" exception=(ex, catch_backtrace())
        end
    finally
        @debug "Finishing external dependency '$(name)'"
        close(channel)
        for channel in values(downstream_neighbours)
            close(channel)
        end
    end
end

function stream_variable(name, stream_output, upstream, downstream, deps, postprocessors;
                         max_train_latency::Integer=20)
    # Initialize the scratch space
    scratch = Dict{String, Any}()

    matcher = Trainmatcher((k for (k, v) in upstream if v isa Union{RemoteChannel, Channel}), max_train_latency)
    matched_trains = Dict{Int, Any}()
    args = Vector{Any}(undef, length(deps))

    # Parameter deps are constant config (e.g. an unassigned optional group
    # dependency), not per-train data, so they never gate execution.
    optional_args = Bool[dep isa Parameter for (_, dep) in deps]

    # We read inputs asynchronously into a single Channel so as not to lose
    # trains from a quickly updating source while waiting for train data from a
    # slowly updating source.
    input_data = Channel{VariableData}(max(4, length(upstream) * 4))
    input_task = Threads.@spawn :interactive try
        @sync for arg in values(upstream)
            if arg isa Union{RemoteChannel, Channel}
                Threads.@spawn :interactive try
                    while isopen(arg) || isready(arg)
                        put!(input_data, take!(arg))
                    end
                catch ex
                    if !(ex isa InvalidStateException)
                        @error "Upstream pump for '$(name)' failed" exception=(ex, catch_backtrace())
                    end
                end
            end
        end
    finally
        close(input_data)
    end

    # Smoothed processing rate (Hz), reported to the client on each output.
    rate = RunningRate()
    try
        while true
            while isempty(matched_trains)
                match_train!(matched_trains, matcher, take!(input_data))
            end

            tid, matched_data = only(matched_trains)
            empty!(matched_trains)

            # Build args from deps, extracting subvariable values as needed
            for (i, (arg_name, dep)) in enumerate(deps)
                if dep isa Parameter
                    args[i] = dep.value
                elseif dep.kind == DepKind_Group
                    args[i] = upstream[dep.name]
                elseif dep.kind == DepKind_Subvariable
                    parent_data = matched_data[dep.parent]
                    subvar = get(parent_data.subvariables, dep.name, nothing)
                    args[i] = isnothing(subvar) ? nothing : subvar.data
                else
                    args[i] = matched_data[dep.name].data
                end
            end

            # Don't execute the variable if any required input is `nothing`
            empty_result = VariableData(tid, name, nothing)
            if any(i -> !optional_args[i] && isnothing(args[i]), eachindex(args))
                putall!(values(downstream), empty_result)
                continue
            end

            # Execute the variable
            f = worker_state.dag_functions[name]
            subvar_values = Dict{String, Any}()
            @debug "Executing variable '$(name)'..."
            @lock worker_state.task_locks[name] try
                out = @with(Meta.tid => tid,
                            Meta.name => name,
                            Meta.scratch => scratch,
                            Meta.subvariables => subvar_values,
                            @invokelatest f(args...))
            catch ex
                @error "Execution of variable '$(name)' failed" exception=(ex, catch_backtrace())
                putall!(values(downstream), empty_result)
                continue
            end

            if !isnothing(out)
                tick!(rate)
            end

            # Run postprocessors on the variable output. Reuse the variable's
            # train-scoped metadata so `tryset` (and anything else reading
            # Meta.*) works the same as inside the variable body.
            if !isempty(postprocessors)
                raw_out = out isa VariableData ? out.data : out
                for (pp_name, pp) in postprocessors
                    try
                        subvar_values[pp_name] = @with(Meta.tid => tid,
                                                       Meta.name => pp_name,
                                                       Meta.scratch => scratch,
                                                       Meta.subvariables => subvar_values,
                                                       pp(raw_out))
                    catch ex
                        @error "Postprocessor '$(pp_name)' for variable '$(name)' failed" exception=(ex, catch_backtrace())
                    end
                end
            end

            # Wrap subvariable values in VariableData
            for (subvar_name, subvar_value) in subvar_values
                subvar_values[subvar_name] = wrap_result(subvar_value, tid, subvar_name)
            end

            # Send output (NaN rate before the second tick → report 0)
            update_rate = isnan(rate.value) ? 0.0 : rate.value
            out = wrap_result(out, tid, name; subvariables=subvar_values, update_rate)
            put!(stream_output, out)
            putall!(values(downstream), out)
            @debug "Pushed output from '$(name)' to: $(keys(downstream))"
        end
    catch ex
        if !(ex isa InvalidStateException)
            @error "Streaming '$(name)' failed" exception=(ex, catch_backtrace())
        end
    finally
        @debug "Finishing variable '$(name)'"

        # Close upstream and downstream channels
        for arg in values(upstream)
            if arg isa Union{RemoteChannel, Channel}
                close(arg)
            end
        end

        for channel in values(downstream)
            close(channel)
        end

        wait(input_task)
    end
end

# Simple function that will asynchronously watch the DAG and close the
# `stream_output` if all variables are finished.
function watch_context(ctx::ContextState)
    while true
        if all(istaskdone.(values(ctx.variable_tasks)))
            close(ctx.stream_output)
            return
        end

        sleep(0.1)
    end
end

"""
    declare_sources(input_name, new_sources)

Notify the system of a new list of sources that can be read from `input`.
An Input should call this function whenever its sources changes.
"""
function declare_sources(input_name, new_sources)
    current_ctx.available_sources[input_name] = new_sources

    pause_pipeline() do
        update_input_sources(current_ctx)
    end
end

function update_input_sources(ctx::ContextState)
    deps = external_dependencies(ctx)
    for (input_name, _) in ctx.inputs
        group = get_input_group(ctx, input_name)
        isnothing(group) && continue

        input_deps = [dep for dep in deps
                      if get(ctx.dep_to_input, string(dep), nothing) == input_name]
        sources = [trainmatcher_dep_string(dep) for dep in input_deps]
        update_sources(group, sources)
    end
end

function pause_pipeline(f::Function)
    foreach(lock, values(worker_state.task_locks))

    try
        f()
    finally
        foreach(unlock, values(worker_state.task_locks))
    end
end

include("offline.jl")

function start_pipeline(ctx::ContextState; offline::Bool=false, input_buffer_size::Int=50)
    ctx.stream_output = variable_channel(offline)
    ctx.events_channel = RemoteChannel(() -> Channel(100))
    ctx.output_forwarder_task = Threads.@spawn :samepool ctx.forwarder(ctx.stream_output)
    errormonitor(ctx.output_forwarder_task)

    global current_ctx = ctx

    # Offline replay registers a single feeder (set up by `run`) as the function
    # backing the "offline" input that replaces all of the context's inputs.
    if !isnothing(ctx.input_feeder)
        worker_state.dag_functions["offline"] = ctx.input_feeder
    end

    # Run one-shot parameter initializers the first time the pipeline starts
    # after loading. Cleared after running so repeat starts don't re-trigger.
    for (name, param) in ctx.parameters
        if !isnothing(param.initializer)
            owner = find_parameter_owner(ctx, name)
            if isnothing(owner)
                param.initializer(param.value)
            else
                param.initializer(owner, param.value)
            end
            param.initializer = nothing
        end
    end

    # Verify all external deps are routed to an input
    unrouted = [string(dep) for dep in external_dependencies(ctx)
                if !haskey(ctx.dep_to_input, string(dep))]
    if !isempty(unrouted)
        throw(XfaContextException("Cannot start pipeline: the following external dependencies have " *
                                  "no input: $(join(unrouted, ", "))"))
    end

    # Start the input functions to feed the DAG
    for (name, deps) in ctx.inputs
        group = nothing
        if !isempty(deps)
            first_arg = first(keys(deps))
            first_dep = deps[first_arg]
            if first_dep isa Dependency && first_dep.kind == DepKind_Group
                group = ctx.groups[first_dep.name]
            else
                error("Couldn't schedule input '$(name)', it has a non-group dependency: $(first_dep)")
            end
        end

        ctx.input_channels[name] = Channel(1)
        ctx.input_tasks[name] = Threads.@spawn input_wrapper(name, group, ctx.input_channels[name])
        errormonitor(ctx.input_tasks[name])
    end

    # Start the input variables, each with downstream channels only for
    # the external deps routed to that input.
    for name in keys(ctx.inputs)
        downstream_neighbours = Dict{String, Union{RemoteChannel, Channel}}()
        for dep in external_dependencies(ctx)
            dep_name = string(dep)
            if get(ctx.dep_to_input, dep_name, nothing) == name
                downstream_neighbours[dep_name] = variable_channel(offline)
            end
        end
        ctx.input_variable_channels[name] = downstream_neighbours

        ctx.input_rates[name] = 0.0
        ctx.input_variables_tasks[name] = Threads.@spawn :samepool stream_input(name, ctx.input_channels[name], downstream_neighbours, ctx.input_rates)
        errormonitor(ctx.input_variables_tasks[name])
    end

    # Start the external dependency variables
    unique_external_deps = external_dependencies(ctx)
    for dep in unique_external_deps
        dep_name = string(dep)
        input_name = ctx.dep_to_input[dep_name]
        input_channel = ctx.input_variable_channels[input_name][dep_name]
        input_neighbour = Neighbour(input_name, input_channel)

        downstream_neighbours = Dict{String, Union{RemoteChannel, Channel}}()
        for neighbour in find_downstream_neighbours(ctx, dep_name, DepKind_Karabo)
            downstream_neighbours[neighbour] = variable_channel(offline)
        end
        ctx.external_dependency_channels[dep_name] = downstream_neighbours

        ctx.external_dependency_tasks[dep_name] = Threads.@spawn :samepool stream_external_dependency(dep_name, input_neighbour, downstream_neighbours)
        errormonitor(ctx.external_dependency_tasks[dep_name])
    end

    # Update the inputs
    update_input_sources(ctx)

    # Offline drops nothing: keep partial trains indefinitely (see Trainmatcher).
    max_train_latency = offline ? -1 : 20

    # Start the variables themselves
    execution_order = topological_sort(ctx)
    for name in execution_order
        # Build up the upstream channel list, deduplicating channels for
        # SubvariableDependency's that share their parent's channel.
        args = OrderedDict{String, Any}()
        for dep in values(ctx.dag[name])
            dep_name = string(dep)

            if dep isa Parameter
                continue
            elseif dep.kind == DepKind_Karabo
                args[dep_name] = ctx.external_dependency_channels[dep_name][name]
            elseif dep.kind == DepKind_Subvariable
                if !haskey(args, dep.parent)
                    args[dep.parent] = ctx.variable_channels[dep.parent][name]
                end
            elseif dep.kind == DepKind_Variable
                args[dep_name] = ctx.variable_channels[dep_name][name]
            elseif dep.kind == DepKind_Group
                args[dep.name] = ctx.groups[dep.name]
            else
                throw(XfaContextException("Unrecognized dependency kind: $(dep.kind)"))
            end
        end

        # Find downstream variables, including those that depend on our
        # subvariables.
        downstream = Dict{String, Union{RemoteChannel, Channel}}()
        for neighbour in find_downstream_neighbours(ctx, name, DepKind_Variable)
            downstream[neighbour] = variable_channel(offline)
        end
        for neighbour in find_downstream_neighbours(ctx, name, DepKind_Subvariable)
            if !haskey(downstream, neighbour)
                downstream[neighbour] = variable_channel(offline)
            end
        end
        ctx.variable_channels[name] = downstream

        # An overridden variable isn't computed: an emitter pushes its override
        # value once per matched train instead of running stream_variable.
        if haskey(ctx.variable_overrides, name)
            ctx.variable_tasks[name] = Threads.@spawn offline_emitter(name, ctx.stream_output, downstream,
                                                                      ctx.variable_overrides[name], ctx.matched_tids)
            errormonitor(ctx.variable_tasks[name])
            continue
        end

        var_pps = Dict{String, AbstractPostprocessor}(
            pp_name => ctx.postprocessors[pp_name]
            for pp_name in get(ctx.variable_postprocessors, name, String[])
        )
        ctx.variable_tasks[name] = Threads.@spawn stream_variable(name, ctx.stream_output, args, downstream, ctx.dag[name], var_pps; max_train_latency)
        errormonitor(ctx.variable_tasks[name])
    end

    # Start the watcher task
    ctx.watcher_task = Threads.@spawn :samepool watch_context(ctx)
    errormonitor(ctx.watcher_task)

    ctx.is_running[] = true

    return nothing
end

function stop_pipeline(ctx::ContextState; timeout=5)
    ctx.is_running[] = false

    for ch in values(ctx.input_channels)
        close(ch)
    end

    # Close the input tasks
    for task in values(ctx.input_tasks)
        wait(task)
    end

    # Close the streaming input tasks
    for outputs in values(ctx.input_variable_channels)
        for channel in values(outputs)
            close(channel)
        end
    end
    for task in values(ctx.input_variables_tasks)
        wait(task)
    end

    # Close the external dependency tasks
    for outputs in values(ctx.external_dependency_channels)
        for channel in values(outputs)
            close(channel)
        end
    end
    for task in values(ctx.external_dependency_tasks)
        wait(task)
    end

    # Close the variables tasks
    for outputs in values(ctx.variable_channels)
        for channel in values(outputs)
            close(channel)
        end
    end
    for task in values(ctx.variable_tasks)
        wait(task)
    end

    wait(ctx.watcher_task)
    if !isnothing(ctx.output_forwarder_task)
        wait(ctx.output_forwarder_task)
    end

    global current_ctx = nothing

    @debug "Pipeline fully stopped"

    return nothing
end

function run(f::Function, ctx::ContextState; timeout=10, kwargs...)
    start_pipeline(ctx; kwargs...)

    task = nothing
    timer = Timer(timeout) do _
        @warn "Function timed out, killing it"
        Threads.@spawn :samepool Base.throwto(task, InterruptException())
    end

    parent_testset = get(task_local_storage(), :__BASETESTNEXT__, [])
    try
        task = Threads.@spawn :samepool begin
            # Set the parent testset so that all @test's get recorded properly
            task_local_storage(:__BASETESTNEXT__, parent_testset)
            f()
        end

        wait(task)
    finally
        close(timer)
        stop_pipeline(ctx)
    end
end

function _is_context_method(m::Method)
    # Check if the method's type parameter belongs to a user context module.
    # For variable/input traits: signature is Tuple{typeof(f), typeof(func)}
    #   where parentmodule(func) is the context module
    # For group traits: signature is Tuple{typeof(f), Type{T}}
    #   where parentmodule(T) is the context module
    length(m.sig.parameters) >= 2 || return false
    T = m.sig.parameters[2]

    owner = if T <: Type
        T.parameters[1]  # extract T from Type{T}
    elseif isdefined(T, :instance)
        T.instance        # extract func from typeof(func)
    else
        return false      # not a concrete singleton (e.g. the default method)
    end

    mod_names = string.(fullname(parentmodule(owner)))
    return any(startswith.(mod_names, "UserContext"))
end

function _cleanup_context_methods()
    for f in (variable_dependencies, variable_subvariables, variable_postprocessors, input_dependencies, group_fields)
        for m in methods(f)
            if _is_context_method(m)
                Base.delete_method(m)
            end
        end
    end
end

function load_from_string(ctx_str::AbstractString; dep_router=Returns(nothing), prelude=Expr[])
    _cleanup_context_methods()

    ctx_module = Module(Symbol(:UserContext, gensym()))
    init_expr = quote
        using XfaContext.NaNStatistics
        using XfaContext.DimensionalData

        using XfaContext
        using XfaContext: VariableData, PlotSpec, LayerSpec, Parameter, Meta
    end
    @eval ctx_module $init_expr

    # Splice in any input namespaces supplied by input-providing packages
    # (e.g. the engine's KaraboInput), so context files can name inputs the
    # pipeline core doesn't itself define.
    for expr in prelude
        @eval ctx_module $expr
    end

    exprs = Expr[]

    # Parse everything
    expr, pos = Base.Meta.parse(ctx_str, 1)
    while expr != nothing
        push!(exprs, expr)
        expr, pos = Base.Meta.parse(ctx_str, pos)
    end

    # Evaluate all exprs
    for expr in exprs
        @eval ctx_module $expr
    end

    @invokelatest load_from_module(ctx_module, exprs; dep_router, prelude)
end

function load_from_module(ctx_module::Module, exprs::Vector{Expr}; dep_router=Returns(nothing), prelude=Expr[])
    parameters = Dict{String, Parameter}()

    # Discover all variables, inputs, group types, and parameters defined
    # in ctx_module by scanning its names and checking for trait methods.
    ctx_variables = Dict{Function, Vector{Any}}()
    ctx_inputs = Dict{Function, Vector{Any}}()
    group_types = Dict{DataType, Group}()

    # Scan ctx_module for variables and parameters
    for name in names(ctx_module; all=true)
        isdefined(ctx_module, name) || continue
        obj = getfield(ctx_module, name)

        if obj isa Parameter
            obj.name = string(name)
            parameters[string(name)] = obj
        elseif obj isa Function && parentmodule(obj) === ctx_module &&
               hasmethod(variable_dependencies, Tuple{typeof(obj)})
            ctx_variables[obj] = variable_dependencies(obj)
        end
    end

    # Collect origin functions that are already represented by context
    # variable references, so we don't add them again as separate variables.
    ctx_origins = Set(variable_origin(f) for f in keys(ctx_variables) if variable_origin(f) !== f)

    # Discover all registered variables, inputs, and group types from trait
    # methods. This includes builtins and variables from included modules.
    all_variables = copy(ctx_variables)
    for m in methods(variable_dependencies)
        F = m.sig.parameters[2]
        isdefined(F, :instance) || continue
        func = F.instance
        if !haskey(all_variables, func) && func ∉ ctx_origins
            all_variables[func] = variable_dependencies(func)
        end
    end

    for m in methods(group_fields)
        T = m.sig.parameters[2].parameters[1]
        group_types[T] = Group(T, Function[])
    end

    for m in methods(input_dependencies)
        F = m.sig.parameters[2]
        func = F.instance
        ctx_inputs[func] = input_dependencies(func)
    end

    # Associate variables with their group types
    for (group_struct, group) in group_types
        for (func, deps) in merge(all_variables, ctx_inputs)
            if !isempty(deps) && deps[1][2] isa Dependency && deps[1][2].kind == DepKind_Group && deps[1][2].group_type == group_struct
                push!(group.variables, func)
            end
        end
    end

    # Check if there are any parameters with the same name as a variable
    ctx_variable_names = string.(nameof.(keys(ctx_variables)))
    common_var_param_names = intersect(ctx_variable_names,
                                       keys(parameters))
    if !isempty(common_var_param_names)
        names_str = join(common_var_param_names, ", ")
        throw(XfaContextException("@Variable's and @Parameter's exist with the same name: $(names_str)"))
    end

    # Look up all the functions that will be called
    functions = Dict{String, Any}()
    for func in keys(ctx_variables)
        functions[string(nameof(func))] = func
    end

    # Check for duplicate variable/input names
    ctx_variable_names = Set(nameof.(keys(ctx_variables)))
    ctx_input_names = Set(nameof.(keys(ctx_inputs)))
    if !isdisjoint(ctx_variable_names, ctx_input_names)
        throw(XfaContextException("Found Variable's and Input's with duplicate names"))
    end

    # Create the DAG (it's just an adjaceny list)
    dag = Dict{String, OrderedDict}()
    for (func, deps) in merge(ctx_variables, ctx_inputs)
        name = nameof(func)

        # If it's a group dependency, we don't schedule it yet. That's done at
        # the end only for the instantiated group structs.
        if length(deps) > 0 && deps[1][2] isa Dependency && deps[1][2].kind == DepKind_Group
            continue
        end

        # Don't add inputs to the DAG, only variables
        if name in ctx_variable_names
            dag[string(name)] = _get_deps(func, parameters)
        end
    end

    # At this point we've added all the non-grouped variables to the DAG and we
    # know which variables belong to which group, so we can schedule all the
    # instantiated groups.
    groups = Dict{String, Any}()
    inputs = Dict{String, Any}()

    # Look at all the top-level names and check if they're groups
    for (group_name, group_type, object) in _get_group_objects(ctx_module, group_types)
        groups[group_name] = object

        # If so, then add all their variables to the DAG
        for variable_func in group_types[group_type].variables
            # If it's an input, handle it later
            if haskey(ctx_inputs, variable_func)
                continue
            end

            dag_deps = _get_deps(variable_func, parameters)

            # Replace the group dependency that originally contained the group
            # type name, with one that names the instantiated group.
            argument_names = collect(keys(dag_deps))
            arg_idx = findfirst(key -> dag_deps[key] == group_dependency(group_type),
                                argument_names)
            dag_deps[argument_names[arg_idx]] = group_dependency(group_name, group_type)

            # Resolve GroupParameter dependencies, which may reference either
            # a Parameter field of the group, another @Variable in the group,
            # or a subvariable of one.
            for (arg_name, dep) in dag_deps
                if dep isa Dependency && dep.kind == DepKind_GroupParameter
                    dot_idx = findfirst('.', dep.parameter)
                    head = isnothing(dot_idx) ? dep.parameter : dep.parameter[1:dot_idx-1]
                    tail = isnothing(dot_idx) ? "" : dep.parameter[dot_idx+1:end]
                    head_sym = Symbol(head)

                    if hasfield(group_type, head_sym) && fieldtype(group_type, head_sym) <: Parameter
                        if !isempty(tail)
                            throw(XfaContextException("Parameter '$head' of group '$(group_name)' cannot be subscripted with '.$tail'"))
                        end
                        param = getproperty(object, head_sym)
                        if !isassigned(param)
                            # Unset optional dependency: kept in the DAG so the
                            # variable's positional args still line up, but the
                            # scheduler ignores it (no channel, no trainmatching)
                            # and stream_variable passes `nothing` for this arg.
                            dag_deps[arg_name] = param
                        elseif !(param.value isa Dependency)
                            throw(XfaContextException("Parameter '$head' of group '$(group_name)' must hold a Dependency value"))
                        else
                            # A dotted name in a DepKind_Variable means a subvariable
                            # reference (the user-facing `Dependency("foo.bar")` form
                            # can't disambiguate). Promote it so topological_sort and
                            # the group-variable rewrite below see the right kind.
                            resolved = param.value
                            if resolved.kind == DepKind_Variable
                                d = findfirst('.', resolved.name)
                                if !isnothing(d)
                                    resolved = subvariable_dependency(resolved.name[1:d-1],
                                                                      resolved.name[d+1:end])
                                end
                            end
                            dag_deps[arg_name] = resolved
                        end
                    elseif any(nameof(f) == head_sym for f in group_types[group_type].variables)
                        var_name = "$group_name.$head"
                        dag_deps[arg_name] = isempty(tail) ? Dependency(var_name) : subvariable_dependency(var_name, tail)
                    else
                        throw(XfaContextException("'$head' is not a Parameter field or @Variable of $(nameof(group_type))"))
                    end
                end
            end

            func_name = string(nameof(variable_func))
            group_var_name = "$group_name.$func_name"
            dag[group_var_name] = dag_deps
            functions[group_var_name] = variable_func

            # Dependencies of the form `foo.bar` are saved as
            # SubvariableDependency's. But these may also refer to groups, so
            # now we go through all the dependencies for all variables and check
            # if any are actually group variables instead of subvariables.
            for var_deps in values(dag)
                for i in eachindex(var_deps)
                    if var_deps[i] == subvariable_dependency(group_name, func_name)
                        var_deps[i] = Dependency(group_var_name)
                    end
                end
            end
        end

        # And add all the parameters too
        # for (param_sym, param_type) in group_types[group_type].parameters
        #     param_name = "$group_name.$param_sym"
        #     parameters[param_name] = Parameter(param_name, getproperty(object, param_sym))
        # end
        for field in fieldnames(group_type)
            if fieldtype(group_type, field) <: Parameter
                param = getproperty(object, field)
                param.name = "$(group_name).$(field)"
                parameters[param.name] = param
            end
        end

        # And all the inputs
        for (input_func, deps) in ctx_inputs
            input_name = string(nameof(input_func))

            if length(deps) == 1
                input_group_type = deps[1][2].group_type
                if input_group_type === group_type
                    group_input_name = "$group_name.$input_name"
                    new_deps = OrderedDict(deps)
                    # Similarly to variables, we replace the group dependency
                    # that contained the group type name with one that contains
                    # the instantiated name.
                    new_deps[first(keys(new_deps))] = group_dependency(group_name, group_type)
                    inputs[group_input_name] = new_deps
                    functions[group_input_name] = input_func
                end
            end
        end
    end

    for (func, deps) in ctx_inputs
        if isempty(deps) || !(deps[1][2] isa Dependency && deps[1][2].kind == DepKind_Group)
            name = string(nameof(func))
            throw(XfaContextException("'$(name)' must belong to a Group to be a valid Input"))
        end
    end

    # Check that it has no cycles by attempting to sort it
    topological_sort(dag)

    ctx_subvariables = Dict{String, Vector{String}}()
    ctx_postprocessors = Dict{String, AbstractPostprocessor}()
    ctx_variable_postprocessors = Dict{String, Vector{String}}()
    for name in keys(dag)
        # For grouped variables the DAG name (e.g. "my_corr.correlate")
        # differs from the bare function name used in the trait methods
        # (e.g. subvariable "correlate.avg"), so we remap.
        func_base = string(nameof(functions[name]))
        ctx_subvariables[name] = [replace(s, func_base => name; count=1)
                                  for s in variable_subvariables(functions[name])]

        pps = variable_postprocessors(functions[name])
        if !isempty(pps)
            ctx_variable_postprocessors[name] = String[]

            for (pp_name, processor) in pps
                pp_name = replace(pp_name, func_base => name; count=1)
                ctx_postprocessors[pp_name] = processor
                push!(ctx_variable_postprocessors[name], pp_name)

                # Add subvariable names that weren't known at macro time
                # (one-arg @add_postprocessor form using default_name)
                if pp_name ∉ ctx_subvariables[name]
                    push!(ctx_subvariables[name], pp_name)
                end

                # Register Parameter fields from postprocessor instances
                for field in fieldnames(typeof(processor))
                    if fieldtype(typeof(processor), field) <: Parameter
                        param = getproperty(processor, field)
                        param.name = "$(pp_name).$(field)"
                        parameters[param.name] = param
                    end
                end
            end
        end
    end

    global worker_state = WorkerState(; dag_functions=functions, current_ctx_module=ctx_module, parameters)
    for name in keys(worker_state.dag_functions)
        worker_state.task_locks[name] = ReentrantLock()
    end

    # Resolve @display references to fully-qualified parameter names. A
    # reference with a dot ("GroupType.field") names a group field and is
    # mapped to the instantiated group: variables in a group are named
    # "$group_name.$func_name", so we use the variable's group_name prefix.
    # A reference without a dot is a top-level Parameter binding name.
    ctx_displays = Dict{String, Vector{String}}()
    for var_name in keys(dag)
        refs = variable_displays(functions[var_name])
        if isempty(refs)
            continue
        end
        resolved = String[]
        for ref in refs
            param_name = if occursin('.', ref)
                _, field = split(ref, '.'; limit=2)
                group_name, _ = split(var_name, '.'; limit=2)
                "$(group_name).$(field)"
            else
                ref
            end
            if !haskey(parameters, param_name)
                throw(XfaContextException("@display references unknown parameter '$(param_name)' (from '$(ref)' on variable '$(var_name)')"))
            end
            push!(resolved, param_name)
        end
        ctx_displays[var_name] = resolved
    end

    ctx = ContextState(; functions, group_types, groups, dag,
                     subvariables=ctx_subvariables,
                     variable_postprocessors=ctx_variable_postprocessors,
                     postprocessors=ctx_postprocessors,
                     displays=ctx_displays,
                     parameters, exprs, inputs, prelude)
    ctx.dep_to_input = build_dep_routing(ctx, dep_router)
    return ctx
end

function load_from_file(ctx_path::AbstractString; dep_router=Returns(nothing), prelude=Expr[])
    if !isfile(ctx_path)
        throw(ArgumentError("$(ctx_path) is not a file!"))
    end

    ctx = load_from_string(read(ctx_path, String); dep_router, prelude)
    ctx.path = ctx_path
    return ctx
end

function _get_group_objects(ctx_module, group_types)
    group_objects = Tuple{String, DataType, Any}[]

    for name in names(ctx_module; all=true)
        object = getproperty(ctx_module, name)
        group_type = findfirst(g -> object isa g.type, group_types)
        if group_type != nothing
            push!(group_objects, (string(name), group_type, object))
        end
    end

    return group_objects
end

function _get_deps(func, parameters)
    final_deps = OrderedDict()

    for (arg_name, dep) in variable_dependencies(func)
        if !(dep isa Dependency)
            throw(ArgumentError("Dependency of type '$(typeof(dep))' is not allowed"))
        end

        if dep.kind == DepKind_Variable && haskey(parameters, dep.name)
            dep = parameters[dep.name]
        end

        final_deps[arg_name] = dep
    end

    return final_deps
end

# Some methods to allow push!'ing to a DimArray
function Base.push!(data::DD.AbstractDimVector, x, lookup_values=nothing)
    push!(DD.parent(data), x)
    for (name, value) in pairs(lookup_values)
        push!(parent(DD.lookup(data, name)), value)
    end

    return DD.rebuild(data)
end

function Base.empty!(data::DD.DimVector)
    empty!(parent(data))
    empty!(parent(DD.lookup(data, DD.dims(data)[1])))

    return DD.rebuild(data)
end

end # module XfaContext
