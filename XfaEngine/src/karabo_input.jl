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

    # Lease TTL (in seconds) reported by the trainmatchers subscribeSources
    # slot. Leases are renewed at half this interval while streaming.
    lease_ttl::Float64 = 30.0

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

    bridge.sources = sources
    if !isempty(sources)
        subscribe_sources(bridge)
    end
end

# Lease the current sources from the trainmatcher through its subscribeSources
# slot. The leases expire after the TTL in the reply, so this must be called
# periodically while streaming to keep the sources alive.
function subscribe_sources(bridge::KaraboInput)
    device = bridge.trainmatcher[]
    reply = call_slot(get_webproxy(device), device.name, "subscribeSources",
                      Dict("sources" => bridge.sources))
    if !reply["success"]
        @warn "Trainmatcher '$(device.name)' rejected the source subscription" reason=get(reply, "reason", "unknown")
    end
    bridge.lease_ttl = reply["ttl"]
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
