# The KaraboInput input group: a pipeline @Input that streams trains from a
# live Karabo bridge over ZMQ. It's an engine-side plugin defined against the
# Context macros/trait functions but kept out of the pipeline core, since it
# depends on the engine's webproxy/device discovery and the ZMQ KaraboBridge
# transport. Context files name it via the prelude registered in XfaEngine.jl.

@Group mutable struct KaraboInput
    manual_configuration::Parameter{Bool} = Parameter(false)
    trainmatcher::Parameter{KaraboDevice}
    address::Parameter{String} = Parameter("")

    sources::Vector{String} = String[]

    # Reusable receive buffers for array payloads, keyed by (source, path).
    # See karabo_bridge.jl BufferRing for the rotation policy.
    buffer_pool::BufferPool = BufferPool()

    # Internal field for testing: when set, get_sources() returns this
    # instead of querying the WebProxy.
    _mock_sources::Union{Vector{String}, Nothing} = nothing
end

Context.input_topic(bridge::KaraboInput) = let t = bridge.trainmatcher[].topic; isempty(t) ? nothing : t end
Context.input_device(bridge::KaraboInput) = let dev = bridge.trainmatcher[]
    isempty(dev.name) ? nothing : dev
end

function Context.get_sources(bridge::KaraboInput)
    if !isnothing(bridge._mock_sources)
        return bridge._mock_sources
    end

    if isnothing(current_engine_state)
        @warn "Engine is not initialized, skipping source discovery for KaraboInput"
        return String[]
    end

    try
        wp = get_webproxy(bridge.trainmatcher[])
        devices = get_devices(wp)
        return collect(keys(devices))
    catch ex
        @error "Failed to get sources from KaraboInput" exception=(ex, catch_backtrace())
        return String[]
    end
end

function Context.update_sources(bridge::KaraboInput, sources)
    if bridge.manual_configuration[]
        @warn "KaraboInput is in manual configuration mode, cannot automatically configure a trainmatcher"
        return
    end

    put_property(bridge.trainmatcher[], "sources", [Dict("source" => s) for s in sources])
end

@Input function stream(bridge::KaraboInput, output)
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
            catch
                break
            end

            # If the channel is full we drop the train data
            if Base.n_avail(bridge_msgs) ≥ bridge_msgs.sz_max
                @warn "Input buffer for $(Context.Meta.name[]) is full, dropping train"
                continue
            else
                put!(bridge_msgs, msg)
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
        while isopen(output)
            sleep(0.1)
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
