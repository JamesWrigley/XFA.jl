# Offline replay against an extra-data DataCollection. Provides the
# `run(ctx, dc; select, override)` method whose pure-Julia machinery lives in
# XfaContext/src/offline.jl.
module XfaContextPythonCallExt

using XfaContext: XfaContext, ContextState, external_dependencies, prepare_offline!,
    run_offline_plan
using DimensionalData: DimArray
import DimensionalData as DD
using PythonCall

# Build the KeyData for a dependency. Control (slow) data is keyed with a
# `.value` suffix, so fall back to that when the bare property isn't a key of
# the source.
function dep_key_data(selection, dep)
    keys = Set(pyconvert(Vector{String}, selection.keys_for_source(dep.source)))
    key = dep.property in keys ? dep.property : dep.property * ".value"
    return selection[pytuple((dep.source, key))]
end

# extra-data labels each per-train array's leading axis "trainId". For a single
# train that axis is a spurious singleton (scalar/slow data), which we drop down
# to the bare value; pulse-resolved data keeps its (frame) axis.
function drop_singleton_train(da)
    if DD.hasdim(da, :trainId) && length(DD.lookup(da, :trainId)) == 1
        squeezed = dropdims(da; dims=DD.dims(da, :trainId))
        return ndims(squeezed) == 0 ? squeezed[] : squeezed
    end
    return da
end

# Feeder that replaces all of the context's inputs: pushes one `(tid, data)` per
# matched train. For each train it pulls every dependency's `.xarray()` and wraps
# it as a DimArray; `copy=false` keeps it a zero-copy view over the Python buffer.
# The GIL is held only while reading a train, not across `put!`, so backpressure
# from the (blocking) pipeline doesn't stall other Python users.
function feed_trains(channel, tids, key_data, deps)
    by_id = PythonCall.GIL.@lock pyimport("extra_data").by_id
    for tid in tids
        data = PythonCall.GIL.@lock begin
            d = Dict{String, Dict{String, Any}}()
            for dep in deps
                da = key_data[dep.name].select_trains(by_id[pylist([tid])]).xarray()
                value = drop_singleton_train(pyconvert(DimArray, da; copy=false))
                get!(d, dep.source, Dict{String, Any}())[dep.property] = value
            end
            d
        end
        put!(channel, (tid, data))
    end
end

# Check whether `dc` is an extra_data DataCollection without importing
# extra_data: if the module isn't already loaded, the object can't be one. This
# mirrors EXtra's `_isinstance_no_import` trick.
function is_data_collection(dc)
    extra_data = pyimport("sys").modules.get("extra_data")
    if pyis(extra_data, pybuiltins.None)
        return false
    end
    return pyisinstance(dc, extra_data.DataCollection)
end

function XfaContext.run(ctx::ContextState, dc::Py;
                        select=String[], override=Dict{String, Any}())
    PythonCall.GIL.@lock begin
        if !is_data_collection(dc)
            throw(ArgumentError("Expected an extra_data DataCollection, got a $(pytype(dc))"))
        end
    end

    plan = copy(ctx)
    prepare_offline!(plan, select, override)

    # Restrict the run to trains where every required (source, property) is
    # present, so the in-engine trainmatcher always sees complete trains. A
    # dependency's proxy part doesn't select extra-data and is ignored.
    deps = external_dependencies(plan)
    selection, tids = PythonCall.GIL.@lock begin
        if isempty(deps)
            dc, pyconvert(Vector{Int}, dc.train_ids)
        else
            sel = dc.select(unique([(dep.source, dep.property) for dep in deps]); require_all=true)
            sel, pyconvert(Vector{Int}, sel.train_ids)
        end
    end

    # Build the per-dependency KeyData up front; the feeder then reads each train
    # from them as it runs.
    key_data = PythonCall.GIL.@lock Dict(dep.name => dep_key_data(selection, dep) for dep in deps)

    # The main thread holds the GIL by default; `GIL.unlock` releases it while
    # the pipeline runs so `feed_trains` (on another task) can lock it per train.
    feeder = channel -> feed_trains(channel, tids, key_data, deps)
    result = run_offline_plan(plan, tids, feeder; wait_guard=PythonCall.GIL.unlock)

    return result
end

end # module XfaContextPythonCallExt
