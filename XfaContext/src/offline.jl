# Offline replay support: running a loaded context against a finite, already
# available dataset (e.g. an extra-data DataCollection) instead of a live input.
# The public entry point is `run(ctx, data_collection; select, override)`, whose
# data-collection method lives in the PythonCall package extension; everything
# here is the pure-Julia machinery it builds on.

# Shallow copy of a context, sharing immutable/loaded state (functions, groups,
# parameters, postprocessors) but with fresh containers for everything `run`
# mutates while building an offline plan, so the loaded context is left pristine
# and can be re-run with different `select`/`override` arguments.
function Base.copy(ctx::ContextState)
    ContextState(; functions=copy(ctx.functions),
                 group_types=ctx.group_types,
                 groups=ctx.groups,
                 dag=copy(ctx.dag),
                 subvariables=copy(ctx.subvariables),
                 variable_postprocessors=copy(ctx.variable_postprocessors),
                 displays=copy(ctx.displays),
                 postprocessors=ctx.postprocessors,
                 parameters=ctx.parameters,
                 exprs=ctx.exprs,
                 inputs=copy(ctx.inputs),
                 prelude=ctx.prelude,
                 dep_to_input=copy(ctx.dep_to_input),
                 path=ctx.path,
                 forwarder=ctx.forwarder,
                 on_parameter_changed=ctx.on_parameter_changed)
end

# Turn a `select` pattern into an anchored regex. Names are dot-separated
# (e.g. "group.var"); `*` is a wildcard, everything else is literal.
function pattern_regex(pattern::AbstractString)
    escaped = replace(pattern, "." => "\\.")
    Regex("^" * replace(escaped, "*" => ".*") * "\$")
end

# The set of variables to keep when filtering `dag` to `roots`: the roots plus
# everything they transitively depend on. Overridden roots have empty deps, so
# the walk naturally stops at them and their upstream is pruned.
function upstream_closure(dag, roots)
    kept = Set{String}(roots)
    queue = collect(roots)
    while !isempty(queue)
        name = pop!(queue)
        haskey(dag, name) || continue
        for dep in values(dag[name])
            if dep isa Dependency && dep.kind in (DepKind_Variable, DepKind_Subvariable)
                producer = dep_variable_name(dep)
                if producer ∉ kept
                    push!(kept, producer)
                    push!(queue, producer)
                end
            end
        end
    end
    return kept
end

# Apply `override` and `select` to an offline plan (a copy of a loaded context).
# Overridden variables have their dependencies cut so they become emitter-driven
# roots; `select` then keeps only the named variables (or subvariables, by their
# parent) plus their transitive upstream, pruning everything else.
function prepare_offline!(plan::ContextState, select, override)
    for (name, value) in override
        if !haskey(plan.dag, name)
            throw(XfaContextException("Cannot override unknown variable '$(name)'"))
        end
        plan.dag[name] = OrderedDict()
        plan.variable_overrides[name] = value
        delete!(plan.variable_postprocessors, name)
        plan.subvariables[name] = String[]
    end

    if !isempty(select)
        roots = Set{String}()
        for pattern in select
            re = pattern_regex(pattern)
            for name in keys(plan.dag)
                if occursin(re, name)
                    push!(roots, name)
                end
            end
            # A pattern may also name a subvariable; keep its parent.
            for (parent, subvars) in plan.subvariables
                if any(sv -> occursin(re, sv), subvars)
                    push!(roots, parent)
                end
            end
        end

        if isempty(roots)
            throw(XfaContextException("`select` matched no variables: $(join(select, ", "))"))
        end

        kept = upstream_closure(plan.dag, roots)
        for name in collect(keys(plan.dag))
            if name ∉ kept
                delete!(plan.dag, name)
                delete!(plan.functions, name)
                delete!(plan.variable_postprocessors, name)
                delete!(plan.subvariables, name)
                delete!(plan.displays, name)
            end
        end
    end

    return plan
end

# Resolve a per-train value. A DimArray carrying a `trainId` dimension is indexed
# per train; anything else is used as-is for every train. Used for both input
# feeds and overrides.
function value_for_train(value, tid)
    if value isa DD.AbstractDimArray && DD.hasdim(value, :trainId)
        slice = @view value[trainId=DD.At(tid)]
        # A trainId-only DimArray slices down to a scalar; higher-dimensional
        # per-train data stays a (zero-copy) view.
        return ndims(slice) == 0 ? slice[] : slice
    end
    return value
end

# The train ids a per-train value provides (a DimArray with a `trainId`
# dimension), or nothing for a constant that broadcasts to every train.
function train_ids(value)
    if value isa DD.AbstractDimArray && DD.hasdim(value, :trainId)
        return collect(DD.lookup(value, :trainId))
    end
    return nothing
end

# Stand-in for stream_variable used by offline overrides: emits the override
# value for each matched train instead of computing the variable. Pacing comes
# from backpressure on the (blocking) downstream channels.
function offline_emitter(name, stream_output, downstream, value, tids)
    try
        for tid in tids
            out = VariableData(tid, name, value_for_train(value, tid))
            put!(stream_output, out)
            putall!(values(downstream), out)
        end
    catch ex
        if !(ex isa InvalidStateException)
            @error "Override emitter for '$(name)' failed" exception=(ex, catch_backtrace())
        end
    finally
        for channel in values(downstream)
            close(channel)
        end
    end
end

# Offline forwarder: drains a finished pipeline's stream_output into `raw`,
# recording each variable's and subvariable's per-train data.
function accumulate_stream!(raw::Dict{String, Vector{Pair{Int, Any}}}, stream)
    try
        while isopen(stream) || isready(stream)
            vd = take!(stream)
            record_sample!(raw, vd.name, vd.tid, vd.data)
            for (sub_name, sub_vd) in vd.subvariables
                record_sample!(raw, sub_name, vd.tid, sub_vd.data)
            end
        end
    catch ex
        if !(ex isa InvalidStateException)
            @error "Offline result collection failed" exception=(ex, catch_backtrace())
        end
    end
end

function record_sample!(raw, name, tid, data)
    if isnothing(name) || isnothing(data)
        return
    end
    push!(get!(raw, name, Pair{Int, Any}[]), tid => data)
end

# Assemble the final per-variable result from accumulated samples, sorted by
# train ID along a `trainId` dimension. Scalar samples become a plain DimVector;
# array samples are stacked into one dense array with `trainId` as the trailing
# dimension (this copies each train's data into a contiguous buffer).
function to_dimarrays(raw::Dict{String, Vector{Pair{Int, Any}}})
    results = Dict{String, DD.AbstractDimArray}()
    for (name, samples) in raw
        sorted = sort(samples; by=first)
        tids = first.(sorted)
        values = [s.second for s in sorted]
        traindim = DD.Dim{:trainId}(tids)
        sample = first(values)
        if sample isa DD.AbstractDimArray
            stacked = stack(DD.parent(v) for v in values)
            results[name] = DD.DimArray(stacked, (DD.dims(sample)..., traindim); name=Symbol(name))
        elseif sample isa AbstractArray
            stacked = stack(values)
            inner = ntuple(i -> DD.Dim{Symbol(:dim_, i)}(), ndims(sample))
            results[name] = DD.DimArray(stacked, (inner..., traindim); name=Symbol(name))
        else
            results[name] = DD.DimArray(values, (traindim,); name=Symbol(name))
        end
    end
    return results
end

# Shared driver for both `run` methods. Given a prepared `plan`, the matched
# train ids and an `input_feeder` closure (pushes `(tid, data)` per train), wire
# the single "offline" input, run the streaming pipeline to completion and
# collect the per-train results. `wait_guard` wraps the blocking wait on the
# pipeline tasks — the PythonCall path passes `GIL.@unlock` so the feeder task
# can take the GIL; the default runs it directly.
function run_offline_plan(plan::ContextState, matched_tids, input_feeder; wait_guard=(f -> f()))
    plan.matched_tids = matched_tids
    plan.input_feeder = input_feeder
    plan.inputs = Dict{String, Any}("offline" => OrderedDict{Any, Any}())
    plan.dep_to_input = build_dep_routing(plan)

    raw = Dict{String, Vector{Pair{Int, Any}}}()
    plan.forwarder = stream -> accumulate_stream!(raw, stream)

    start_pipeline(plan; offline=true)
    try
        wait_guard() do
            wait(plan.watcher_task)
            wait(plan.output_forwarder_task)
        end
    finally
        stop_pipeline(plan)
    end

    return to_dimarrays(raw)
end

# Offline replay entry point. `values` maps each external (Karabo) dependency
# name to the data fed for it: a DimArray with a `trainId` dimension supplies a
# distinct element per train, anything else is sent unchanged on every train.
# `select`/`override` prune and patch the DAG as in `prepare_offline!`.
#
# The matched trains are the intersection of the train ids offered by every
# per-train source (DimArrays in `values` and `override`); constants don't
# constrain it. If nothing carries a `trainId` dimension the stream would be
# infinite, so we error.
#
# A separate method taking an extra-data DataCollection is provided by the
# PythonCall package extension.
function run(ctx::ContextState, values::Dict; select=String[], override=Dict())
    plan = copy(ctx)
    prepare_offline!(plan, select, override)

    tid_sets = [train_ids(value) for value in Iterators.flatten((Base.values(values), Base.values(override)))]
    filter!(!isnothing, tid_sets)
    if isempty(tid_sets)
        throw(ArgumentError("None of `values` or `override` carry a trainId dimension; the offline stream would be infinite"))
    end
    matched = sort(collect(intersect(Set.(tid_sets)...)))

    deps = external_dependencies(plan)
    feeder = function (channel)
        for tid in matched
            data = Dict{String, Dict{String, Any}}()
            for dep in deps
                haskey(values, dep.name) || continue
                get!(data, dep.source, Dict{String, Any}())[dep.property] = value_for_train(values[dep.name], tid)
            end
            put!(channel, (tid, data))
        end
    end

    return run_offline_plan(plan, matched, feeder)
end
