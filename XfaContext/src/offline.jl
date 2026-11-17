# Offline replay support: running a loaded context against a finite, already
# available dataset (e.g. an extra-data DataCollection) instead of a live input.
# The public entry point is `run(ctx, data_collection; select, override)`, whose
# data-collection method lives in the PythonCall package extension; everything
# here is the pure-Julia machinery it builds on.

const PythonCall_pkgid = Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"), "PythonCall")

# Load PythonCall on first use (like Pkg in IJulia). Callers must
# `@invokelatest` extension methods.
#
# The importing thread becomes Python's main thread, and finalising Python at
# exit only works from that thread: from any other one, a finaliser (e.g.
# h5py's) re-taking the GIL under another thread state is parked forever. Exit
# hooks run on thread 1, so the import is pinned there.
function load_pythoncall()
    guard_pythoncall_import(() -> Base.require(PythonCall_pkgid))
    return Base.loaded_modules[PythonCall_pkgid]
end

# Run `f`, which may import PythonCall (directly or through another package),
# on thread 1 if PythonCall isn't loaded yet.
function guard_pythoncall_import(f)
    if haskey(Base.loaded_modules, PythonCall_pkgid) || Threads.threadid() == 1
        return f()
    else
        # jl_set_task_tid takes a 0-based thread id, so 0 is thread 1
        task = Task(f)
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, 0)
        schedule(task)
        return fetch(task)
    end
end

# Offline input API over an extra-data DataCollection, implemented by the
# PythonCall extension.
function open_data_collection end
function data_collection_devices end
function data_collection_schema end
function data_collection_run end
function open_stream end
function feed! end

# Replace Python-backed arrays with Julia views over the same memory, the
# caller must `GC.@preserve` the original.
unwrap_python(x) = x

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
                 on_output=ctx.on_output,
                 on_parameter_changed=ctx.on_parameter_changed,
                 on_displayable_changed=ctx.on_displayable_changed)
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

    kept
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
                for target in [name; plan.subvariables[name]]
                    delete!(plan.displays, target)
                end
                delete!(plan.subvariables, name)
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

# Shared driver for both `run` methods. Given a prepared `plan`, the matched
# train ids and an `input_feeder` (called with a channel to push `(tid, data)`
# per train, and `release!`d once a train is done), wire the single "offline"
# input, run the streaming pipeline to completion and collect the per-train
# results.
function run_offline_plan(plan::ContextState, matched_tids, input_feeder)
    plan.matched_tids = matched_tids
    plan.input_feeder = input_feeder
    plan.inputs = Dict{String, Any}("offline" => OrderedDict{Any, Any}())
    plan.dep_to_input = build_dep_routing(plan)

    # Every output is recorded per train, in a run without boundaries that's
    # never finalised.
    plan.saved_outputs = Dict{String, Maybe{SaveMode}}(name => SaveMode_PerTrain for name in output_names(plan))
    plan.damnit_enabled[] = true
    session = RunSession(; proposal=0, run=0, first_tid=-1)
    plan.run_session = Base.Lockable(Ref{Maybe{RunSession}}(session))

    start_pipeline(plan; offline=true)
    try
        wait(plan.watcher_task)
        wait(plan.output_forwarder_task)
    finally
        stop_pipeline(plan)
    end

    return Dict{String, DD.AbstractDimArray}(name => saved_value(name, buffer) for (name, buffer) in session.buffers)
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
