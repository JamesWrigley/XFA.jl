# The KaraboInput input group: a pipeline @Input that streams trains from a
# live Karabo bridge over ZMQ. It's an engine-side plugin defined against the
# Context macros/trait functions but kept out of the pipeline core, since it
# depends on the engine's webproxy/device discovery and the ZMQ KaraboBridge
# transport. Context files name it via the prelude registered in XfaEngine.jl.

@Group mutable struct KaraboInput
    manual_configuration::Parameter{Bool} = Parameter(false)
    trainmatcher::Parameter{KaraboDevice} = Parameter{KaraboDevice}(; optional=true)
    address::Parameter{String} = Parameter("")

    offline::Parameter{Bool} = Parameter{Bool}(; value=false, update_handler=run_changed)
    proposal::Parameter{Int} = Parameter{Int}(; value=0, update_handler=run_changed)
    run::Parameter{Int} = Parameter{Int}(; value=0, update_handler=run_changed)
    run_directory::Parameter{String} = Parameter{String}(; value="", update_handler=run_changed)
    rate::Parameter{Float64} = Parameter(10.0)

    progress::Displayable{Tuple{Int, Int}} = Displayable((0, 0))
    processed::Int = 0
    progress_sent::Float64 = 0.0

    sources::Vector{String} = String[]
    deps::Vector{Context.Dependency} = Context.Dependency[]

    dc::Any = nothing
    dc_key::Tuple{Int, Int, String} = (0, 0, "")
    stream::Any = nothing

    # Lease TTL (in seconds) reported by the trainmatchers subscribeSources
    # slot. Leases are renewed at half this interval while streaming.
    lease_ttl::Float64 = 30.0

    # Reusable receive buffers for array payloads, keyed by (source, path).
    # See karabo_bridge.jl BufferRing for the rotation policy.
    buffer_pool::BufferPool = BufferPool()

    # Internal field for testing: when set, get_sources() returns this
    # instead of querying the WebProxy.
    _mock_sources::Union{Vector{SourceInfo}, Nothing} = nothing
end

# Opens the run before the change is acknowledged, so the GUI keeps the node
# disabled until it's loaded.
function run_changed(bridge::KaraboInput, _)
    if bridge.offline[]
        data_collection(bridge)
    end
    if Context.current_ctx.is_running[]
        Context.request_rewire(Context.current_ctx)
    end
end

function data_collection(bridge::KaraboInput)
    key = (bridge.proposal[], bridge.run[], bridge.run_directory[])
    if key == (0, 0, "")
        return nothing
    elseif isnothing(bridge.dc) || bridge.dc_key != key
        Context.load_pythoncall()
        @info "Opening run for KaraboInput" proposal=key[1] run=key[2] directory=key[3]
        bridge.dc = @invokelatest Context.open_data_collection(key...)
        bridge.dc_key = key
    end
    return bridge.dc
end

# Files don't record a topic, but by convention it's the first word of the
# device name, e.g. "MID" for "MID_FOO_BAR/BAZ/QUUX".
function source_topic(name)
    device = first(split(name, '/'))
    idx = findfirst('_', device)
    if isnothing(idx) || idx == 1
        return "offline"
    else
        return device[1:idx - 1]
    end
end

function Context.input_topic(bridge::KaraboInput)
    if !bridge.offline[] && isassigned(bridge.trainmatcher) && !isempty(bridge.trainmatcher[].topic)
        return bridge.trainmatcher[].topic
    else
        return nothing
    end
end

function Context.input_device(bridge::KaraboInput)
    if !bridge.offline[] && isassigned(bridge.trainmatcher) && !isempty(bridge.trainmatcher[].name)
        return bridge.trainmatcher[]
    else
        return nothing
    end
end

function Context.get_sources(bridge::KaraboInput)
    if !isnothing(bridge._mock_sources)
        return bridge._mock_sources
    end

    try
        if bridge.offline[]
            dc = data_collection(bridge)
            if isnothing(dc)
                return SourceInfo[]
            end
            devices = @invokelatest Context.data_collection_devices(dc)
            return [SourceInfo(source_topic(name), name, class_id) for (name, class_id) in devices]
        elseif isnothing(current_engine_state)
            @warn "Engine is not initialized, skipping source discovery for KaraboInput"
            return SourceInfo[]
        elseif !isassigned(bridge.trainmatcher)
            return SourceInfo[]
        else
            topic = bridge.trainmatcher[].topic
            wp = get_webproxy(bridge.trainmatcher[])
            devices = get_devices(wp)
            return [SourceInfo(topic, name, info["classId"]) for (name, info) in devices]
        end
    catch ex
        @error "Failed to get sources from KaraboInput" exception=(ex, catch_backtrace())
        return SourceInfo[]
    end
end

function Context.update_sources(bridge::KaraboInput, deps)
    bridge.deps = deps
    if bridge.offline[]
        return
    elseif bridge.manual_configuration[]
        @warn "KaraboInput is in manual configuration mode, cannot automatically configure a trainmatcher"
        return
    end

    bridge.sources = [Context.trainmatcher_dep_string(dep) for dep in deps]
    if !isempty(bridge.sources)
        subscribe_sources(bridge)
    end
end

function release_train(bridge::KaraboInput, tid)
    stream = bridge.stream
    if !isnothing(stream)
        @invokelatest Context.release!(stream, tid)
    end

    total = bridge.progress[][2]
    if bridge.offline[] && total > 0
        bridge.processed += 1
        if bridge.processed == total || time() - bridge.progress_sent > 1
            bridge.progress_sent = time()
            bridge.progress[] = (bridge.processed, total)
        end
    end
end

function stream_run(bridge::KaraboInput, output)
    Context.declare_sources(Context.Meta.name[], Context.get_sources(bridge))
    dc = data_collection(bridge)
    if isnothing(dc)
        error("No run configured for KaraboInput, set proposal/run or run_directory")
    end

    monitored = intersect(bridge.deps, Context.monitored_dependencies(Context.current_ctx))
    bridge.stream = @invokelatest Context.open_stream(dc, bridge.deps, monitored)
    bridge.processed = 0
    bridge.progress_sent = 0.0
    bridge.progress[] = (0, length(bridge.stream.train_ids))
    try
        @invokelatest Context.feed!(bridge.stream, output, bridge.rate)
    finally
        stream = bridge.stream
        bridge.stream = nothing
        @invokelatest close(stream)
    end
end

# Lease the current sources from the trainmatcher through its subscribeSources
# slot. The leases expire after the TTL in the reply, so this must be called
# periodically while streaming to keep the sources alive.
function subscribe_sources(bridge::KaraboInput)
    device = bridge.trainmatcher[]
    reply = call_slot(get_webproxy(device), device.name, "subscribeSources",
                      Dict("sources" => bridge.sources); timeout=15)
    if !reply["success"]
        @warn "Trainmatcher '$(device.name)' rejected the source subscription" reason=get(reply, "reason", "unknown")
    end
    bridge.lease_ttl = reply["ttl"]
end

@Input function stream(bridge::KaraboInput, output)
    if bridge.offline[]
        stream_run(bridge, output)
    else
        stream_bridge(bridge, output)
    end
end

function stream_bridge(bridge::KaraboInput, output)
    if !bridge.manual_configuration[]
        Context.declare_sources(Context.Meta.name[], Context.get_sources(bridge))

        # If no address is set, pick the first available one from the trainmatcher
        if isempty(bridge.address[])
            config = get_config(bridge.trainmatcher[])
            outputs = config["zmqOutputs"]
            isempty(outputs) && error("No ZMQ outputs available from trainmatcher")
            bridge.address.value = outputs[1]["address"]

            # Notify clients of the new address
            engine_state = current_engine_state
            if !isnothing(engine_state)
                for client in values(engine_state.clients)
                    Protocol.server_send(client.websocket, ParameterChanged(bridge.address))
                end
            end
        end
    end

    if isempty(bridge.address[])
        error("No address configured for KaraboInput")
    end
    client = KaraboBridgeClient(bridge.address[])

    # Start a task just to read from the bridge. Note that this is separate from
    # the task to put it into the output channel to avoid a race condition where
    # the output channel is closed while we're stuck waiting for the next bridge
    # message.
    bridge_msgs = Channel(10)
    input_task = Threads.@spawn :samepool try
        while isopen(bridge_msgs)
            # take!() may throw an exception when the client is closed
            local msg
            try
                msg = take!(client, bridge.buffer_pool)
            catch ex
                if !isopen(client.socket)
                    break
                else
                    @error "Failed to read a train from the Karabo bridge, skipping it" exception=(ex, catch_backtrace())
                    continue
                end
            end

            # If the channel is full we drop the train data
            if Base.n_avail(bridge_msgs) ≥ bridge_msgs.sz_max
                @warn "Input buffer for $(Context.Meta.name[]) is full, dropping train"
                continue
            else
                # put!() may throw when the channel is closed concurrently
                # during shutdown
                try
                    put!(bridge_msgs, msg)
                catch ex
                    if !(ex isa InvalidStateException)
                        @error "KaraboInput could not push to output channel" exception=(ex, catch_backtrace())
                    end

                    break
                end
            end
        end
    finally
        close(output)
    end
    errormonitor(input_task)
    bind(bridge_msgs, input_task)

    output_task = Threads.@spawn :samepool for msg in bridge_msgs
        data, metadata = msg
        tid = first(values(metadata))["timestamp.tid"]

        # put!() may throw when the channel is closed
        try
            put!(output, (tid, data))
        catch
        end
    end

    try
        last_renewal = time()
        while isopen(output)
            sleep(0.1)

            # Renew the source leases at half the TTL so that a single slow
            # or failed call doesn't let them expire. A failed renewal is only
            # logged since transient webproxy errors shouldn't stop streaming.
            if !bridge.manual_configuration[] && !isempty(bridge.sources) &&
               time() - last_renewal > bridge.lease_ttl / 2
                try
                    subscribe_sources(bridge)
                catch ex
                    @warn "Failed to renew the trainmatcher source leases" exception=ex
                end
                last_renewal = time()
            end
        end
    catch ex
        if !(ex isa InvalidStateException)
            rethrow()
        end
    finally
        close(client)
        close(bridge_msgs)
        wait(input_task)
        wait(output_task)
    end
end
