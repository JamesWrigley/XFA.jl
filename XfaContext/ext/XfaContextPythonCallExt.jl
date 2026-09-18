# Offline replay against an extra-data DataCollection: the stream API declared
# in XfaContext/src/offline.jl and the `run(ctx, dc; select, override)` method
# built on it.
module XfaContextPythonCallExt

using XfaContext: XfaContext, ContextState, Dependency, external_dependencies,
    monitored_dependencies, prepare_offline!, run_offline_plan
using DimensionalData: DimArray
import DimensionalData as DD
using FileWatching: FDWatcher
using PythonCall

# Run `expr` holding the GIL with the task pinned to its OS thread, since
# releasing the GIL from another thread than the one that took it crashes.
# Keep Julia blocking primitives (`put!`, `wait`) outside it.
macro pysafe(expr)
    quote
        task = current_task()
        was_sticky = task.sticky
        task.sticky = true
        state = PythonCall.C.PyGILState_Ensure()
        try
            $(esc(expr))
        finally
            PythonCall.C.PyGILState_Release(state)
            task.sticky = was_sticky
        end
    end
end

# Build the KeyData for a dependency. Control (slow) data is keyed with a
# `.value` suffix, so fall back to that when the bare property isn't a key of
# the source.
function dep_key_data(selection, dep)
    keys = Set(pyconvert(Vector{String}, selection.keys_for_source(dep.source)))
    key = dep.property in keys ? dep.property : dep.property * ".value"
    return selection[pytuple((dep.source, key))]
end

# Label each distinct (source, property) for the streamer, which hands a train's
# data back as fields of a dataclass, so the labels have to be identifiers
# rather than dependency names. Returns the sources to stream and the label each
# dependency reads its data from; two dependencies naming the same data (one of
# them through a proxy, say) share a label so it's only read once.
function label_sources(selection, deps)
    sources = pydict()
    labels = Dict{String, Symbol}()
    by_data = Dict{Tuple{String, String}, Symbol}()

    for dep in deps
        data = (dep.source, dep.property)
        if !haskey(by_data, data)
            label = Symbol("source_", length(by_data) + 1)
            by_data[data] = label
            sources[string(label)] = dep_key_data(selection, dep)
        end

        labels[dep.name] = by_data[data]
    end

    return sources, labels
end

# Showing a Python exception needs the GIL, which the pipeline doesn't hold when
# it logs a failed input, so re-raise it as a plain error rendered under the GIL.
function pyguard(f)
    try
        f()
    catch ex
        if ex isa PyException
            throw(ErrorException(@pysafe sprint(showerror, ex)))
        end
        rethrow()
    end
end

function XfaContext.open_data_collection(proposal::Integer, run::Integer, directory::AbstractString)
    pyguard() do
        @pysafe begin
            extra_data = pyimport("extra_data")
            if isempty(directory)
                extra_data.open_run(proposal, run)
            else
                extra_data.RunDirectory(directory)
            end
        end
    end
end

# `(device, class_id)` pairs like a webproxy lists devices: instrument sources
# `DEV:pipe` collapse to their device.
function XfaContext.data_collection_devices(dc::Py)
    pyguard() do
        @pysafe begin
            devices = Dict{String, String}()
            for src in dc.control_sources
                class_id = dc[src].device_class
                devices[pyconvert(String, src)] = pyis(class_id, pybuiltins.None) ? "" : pyconvert(String, class_id)
            end
            for src in dc.instrument_sources
                get!(devices, first(split(pyconvert(String, src), ':')), "")
            end
            return sort!([(name, class_id) for (name, class_id) in devices])
        end
    end
end

function insert_leaf!(node, key)
    parts = split(key, '.')
    for part in parts[1:end - 1]
        node = get!(node, String(part), Dict{String, Any}())
    end
    node[String(parts[end])] = Dict{String, Any}("nodeType" => "Leaf")
end

# A Karabo-style schema for a device in the run, in the shape the client's
# `schema_property_names` parses.
function XfaContext.data_collection_schema(dc::Py, device::AbstractString)
    pyguard() do
        @pysafe begin
            schema = Dict{String, Any}()
            if pyconvert(Bool, device in dc.control_sources)
                for key in dc[device].keys(; inc_timestamps=false)
                    insert_leaf!(schema, pyconvert(String, key))
                end
            end

            for src in dc.instrument_sources
                dev, pipe = split(pyconvert(String, src), ':'; limit=2)
                if dev == device
                    pipe_schema = Dict{String, Any}()
                    for key in dc[src].keys()
                        insert_leaf!(pipe_schema, pyconvert(String, key))
                    end
                    schema[String(pipe)] = Dict{String, Any}("noInputShared" => true, "schema" => pipe_schema)
                end
            end
            return schema
        end
    end
end

function XfaContext.unwrap_python(array::PyArray{T, N, M, L, T}) where {T, N, M, L}
    if strides(array) != Base.size_to_strides(1, size(array)...)
        throw(ArgumentError("Cannot unwrap a non-contiguous PyArray"))
    end
    return unsafe_wrap(Array, array.ptr, size(array))
end
python_backed(array::PyArray) = true
python_backed(array::AbstractArray) = parent(array) !== array && python_backed(parent(array))

function XfaContext.unwrap_python(array::AbstractArray)
    if python_backed(array)
        throw(ArgumentError("Cannot unwrap a Python-backed $(nameof(typeof(array)))"))
    end
    return array
end

function XfaContext.unwrap_python(array::SubArray)
    if python_backed(array)
        return view(XfaContext.unwrap_python(parent(array)), parentindices(array)...)
    else
        return array
    end
end

function XfaContext.unwrap_python(value::DimArray)
    if python_backed(value) || any(dim -> python_backed(parent(DD.lookup(dim))), DD.dims(value))
        dims = map(DD.dims(value)) do dim
            lookup = DD.lookup(dim)
            DD.rebuild(dim, DD.rebuild(lookup; data=XfaContext.unwrap_python(parent(lookup))))
        end
        return DD.rebuild(value; data=XfaContext.unwrap_python(parent(value)), dims)
    else
        return value
    end
end

# A view into the streamer's buffers, valid until the train is returned. A
# scalar (slow) property arrives as a 0-dimensional array, unwrapped here.
function train_value(array)
    value = pyconvert(DimArray, array; copy=false)
    return ndims(value) == 0 ? value[] : value
end

# A run replayed train by train. `monitored` is the subset of `deps` that
# groups watch for changes. Train objects stay in `pending` until `release!`
# hands their slots back. Without deps the stream only supplies the train clock.
struct TrainStream
    streamer::Union{Py, Nothing}
    iterator::Union{Py, Nothing}
    deps::Vector{Dependency}
    monitored::Vector{Dependency}
    labels::Dict{String, Symbol}
    train_ids::Vector{Int}
    pending::Base.Lockable{Dict{Int, Py}, ReentrantLock}
    last_values::Dict{String, Any}
    last_update_tids::Dict{String, Int}
end

# Only trains where every (source, property) is present are streamed. The
# streamer reads up to `buffer_slots` chunks of `trains_per_chunk` trains ahead.
function XfaContext.open_stream(dc::Py, deps::Vector{Dependency}, monitored::Vector{Dependency}=Dependency[];
                                trains_per_chunk=2, buffer_slots=8)
    pyguard() do
        @pysafe begin
            if isempty(deps)
                train_ids = pyconvert(Vector{Int}, dc.train_ids)
                return TrainStream(nothing, nothing, deps, monitored, Dict{String, Symbol}(), train_ids,
                                   Base.Lockable(Dict{Int, Py}()), Dict{String, Any}(), Dict{String, Int}())
            end

            selection = dc.select(unique([(dep.source, dep.property) for dep in deps]); require_all=true)
            train_ids = pyconvert(Vector{Int}, selection.train_ids)
            sources, labels = label_sources(selection, deps)
            streamer = pyimport("extra_data.export").MemoryStreamer()
            iterator = streamer.event_iterator(sources; trains_per_part=1,
                                               reader_trains_per_chunk=trains_per_chunk,
                                               n_buffer_slots=buffer_slots)
            return TrainStream(streamer, iterator, deps, monitored, labels, train_ids,
                               Base.Lockable(Dict{Int, Py}()), Dict{String, Any}(), Dict{String, Int}())
        end
    end
end

# Convert a train object to the `(tid, data)` the input emits. Files have no
# update tid for control properties, so the `<property>.timestamp.tid` stamp
# `stream_input` checks is synthesized: bumped whenever the value changes.
function train_data(stream::TrainStream, td)
    tid = pyconvert(Int, td.trainId)
    data = Dict{String, Dict{String, Any}}()
    for dep in stream.deps
        value = train_value(getproperty(td, stream.labels[dep.name]))
        get!(data, dep.source, Dict{String, Any}())[dep.property] = value
    end

    for dep in stream.monitored
        value = data[dep.source][dep.property]
        if !haskey(stream.last_values, dep.name) || !isequal(stream.last_values[dep.name], value)
            stream.last_values[dep.name] = value isa AbstractArray ? copy(value) : value
            stream.last_update_tids[dep.name] = tid
        end
        data[dep.source][dep.property * ".value"] = value
        data[dep.source][dep.property * ".timestamp.tid"] = stream.last_update_tids[dep.name]
    end

    return tid, data
end

# Push one `(tid, data)` per train into `channel` until the stream ends or the
# channel is closed, at most `rate[]` trains per second (read per train, so it
# can change live). The iterator's fd is only a (possibly spurious) wake-up
# call, so after each one every ready train is drained.
function XfaContext.feed!(stream::TrainStream, channel, rate=Ref(Inf))
    next = time()
    if isnothing(stream.iterator)
        for tid in stream.train_ids
            wait_time = next - time()
            if wait_time > 0
                sleep(wait_time)
            end
            next = time() + 1 / rate[]
            put!(channel, (tid, Dict{String, Dict{String, Any}}()))
        end
        return
    end

    pyguard() do
        iterator = stream.iterator
        fd = @pysafe pyconvert(Int, iterator.fd)
        watcher = FDWatcher(RawFD(fd), true, false)

        try
            while true
                wait(watcher)

                while true
                    train = @pysafe begin
                        td = iterator.get()
                        pyis(td, pybuiltins.None) ? nothing : (train_data(stream, td)..., td)
                    end

                    if isnothing(train)
                        break
                    end

                    tid, data, td = train
                    @lock stream.pending stream.pending[][tid] = td
                    wait_time = next - time()
                    if wait_time > 0
                        sleep(wait_time)
                    end
                    next = time() + 1 / rate[]
                    put!(channel, (tid, data))
                end

                if @pysafe pyconvert(Bool, iterator.ended)
                    break
                end
            end
        catch ex
            if !(ex isa InvalidStateException)
                rethrow()
            end
        finally
            close(watcher)
        end
    end
end

function XfaContext.release!(stream::TrainStream, tid::Integer)
    if !isnothing(stream.iterator)
        td = @lock stream.pending pop!(stream.pending[], tid)
        pyguard() do
            @pysafe stream.iterator.done(td)
        end
    end
end

function Base.close(stream::TrainStream)
    if !isnothing(stream.iterator)
        pyguard() do
            @pysafe begin
                try
                    stream.iterator.close()
                finally
                    stream.streamer.close()
                end
            end
        end
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
                        select=String[], override=Dict{String, Any}(),
                        trains_per_chunk=2, buffer_slots=8)
    @pysafe begin
        if !is_data_collection(dc)
            throw(ArgumentError("Expected an extra_data DataCollection, got a $(pytype(dc))"))
        end
    end

    plan = copy(ctx)
    prepare_offline!(plan, select, override)

    deps = external_dependencies(plan)
    monitored = intersect(deps, monitored_dependencies(plan))
    stream = XfaContext.open_stream(dc, deps, monitored; trains_per_chunk, buffer_slots)
    plan.on_train_processed = tid -> XfaContext.release!(stream, tid)

    # Release the main thread's GIL so the feeder and the post-pipeline stage
    # can take it per train.
    return PythonCall.GIL.unlock() do
        try
            run_offline_plan(plan, stream.train_ids, channel -> XfaContext.feed!(stream, channel))
        finally
            close(stream)
        end
    end
end

end # module XfaContextPythonCallExt
