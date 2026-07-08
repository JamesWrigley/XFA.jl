module XfaEngine

# The pipeline core. `Context` is kept as a back-compat alias for the module so
# existing `Context.foo` references (here and in macro-generated code) resolve.
using XfaContext
using XfaContext: ContextState, VariableData, ArrayMetadata, KaraboDevice, @Group, @Input, Parameter
const Context = XfaContext

include("zfp_workspace.jl")
include("karabo_bridge.jl")

import TOML
import Glob
include("settings.jl")
include("protocol.jl")

using .KaraboBridge: KaraboBridgeClient, BufferPool
include("webproxy.jl")

import TOML
import Sockets
import Sockets: listen, close, @ip_str, TCPServer
import DistributedNext: @everywhere, @fetchfrom, workers, procs
using Serialization

import HTTP
import HTTP: WebSockets
import Revise
import RemoteREPL
using DimensionalData: DimArray

using .Protocol
using .ZfpWorkspaces: ZfpWorkspace, CompressedArray, compress_array, should_compress
using Accessors: @set

# The KaraboInput @Group/@Input lives here (not in Context) because it depends
# on the engine's webproxy/device discovery and the ZMQ KaraboBridge transport.
include("karabo_input.jl")


"""Find the closest available port to `port_hint`."""
function getavailableport(port_hint; interface=ip"0.0.0.0")
    port_range_end = min(65535, port_hint + 5000)
    available_port = -1

    for port in port_hint:port_range_end
        try
            s = listen(interface, port)
            close(s)
            return port
        catch
            continue
        end
    end

    error("Could not find an available port between $(port_hint) and $(port_range_end)")
end

# function heartbeat(state::ClientState)
#     while !WebSockets.isclosed(state.websocket)
#         WebSockets.send(state.websocket, "ping")
#         sleep(5)
#     end
# end

@kwdef mutable struct ClientState
    websocket::WebSockets.WebSocket
    last_heartbeat::Float64
    handler_task::Union{Task, Nothing} = nothing
    # heartbeat_task::Task

    # Fully-qualified names of array-valued variables this client wants
    # forwarded, mapped to the requested zfp precision (-1 for default).
    # Updated via `SetVariableSubscriptions`.
    subscriptions::Dict{String, Int} = Dict{String, Int}()
end

@kwdef mutable struct EngineState
    websocket_port::Int = -1
    karabo_bridge_port::Int = -1
    websocket_listener_task::Union{Task, Nothing} = nothing
    clients::Dict{String, ClientState} = Dict()

    webproxies::Dict{String, WebProxy} = Dict()
    routing_rules::Vector{RoutingRule} = RoutingRule[]
    remap_rules::Vector{RemapRule} = copy(BUILTIN_REMAP_RULES)

    remoterepl_server::TCPServer = TCPServer()
    remoterepl_task::Union{Task, Nothing} = nothing

    ctx::ContextState = ContextState()

    # One zfp workspace per qualified variable name. Sized to the variable's
    # data on first use and reused across trains; switching precision on the
    # same variable just runs zfp again over the same buffers.
    zfp_workspaces::Dict{String, ZfpWorkspace} = Dict{String, ZfpWorkspace}()

    channel_stats_task::Union{Task, Nothing} = nothing

    stop_event::Base.Event = Base.Event()
    stop_task::Union{Task, Nothing} = nothing
end

current_engine_state::Union{EngineState, Nothing} = nothing

# 0-D values (numbers, strings, scalar arrays) are cheap to ship and always
# forwarded; multi-element arrays must be opted into via subscription.
is_scalar_data(x) = !(x isa AbstractArray) || ndims(x) == 0

# Whether a variable's array is eligible for lossy compression. Extends the
# array-level check with a plotting exception: a 2D array with a color-bound
# plot layer is rendered as a bunch of lines, whose fine per-line detail lossy
# compression mangles, so it's compressed losslessly instead (see client_view_for).
function ZfpWorkspaces.should_compress(variable::VariableData)
    data = variable.data
    plotted_as_lines = data isa AbstractMatrix &&
        any(s -> any(l -> !isnothing(l.color), s.layers), variable.plot_specs)
    return should_compress(data) && !plotted_as_lines
end

# Sentinel precision used to cache the metadata-only view of a non-subscribed
# array payload. Real precisions are >= 0 (and -1 maps to the per-eltype
# default inside compress_array), so this never collides with a real client request.
const METADATA_PRECISION = typemin(Int)

# Build (or fetch from `cache`) the per-client view of a single (sub)variable:
# scalars pass through, non-subscribed arrays become ArrayMetadata, subscribed
# compressible arrays become a CompressedArray at the requested precision, and
# subscribed non-compressible arrays pass through raw. The cache stores at most
# one (precision, VariableData) per qualified name; if a client asks for a
# different precision than the cached one we just recompress and overwrite.
function client_view_for(state::EngineState, variable::VariableData, qualified::String,
                         subscriptions::Dict{String, Int},
                         cache::Dict{String, Tuple{Int, VariableData}})
    data = variable.data
    requested = get(subscriptions, qualified, nothing)

    precision = if is_scalar_data(data)
        nothing
    elseif isnothing(requested)
        METADATA_PRECISION
    elseif should_compress(data)
        # Compress losslessly when the client didn't opt into lossy compression,
        # or when the array must stay exact because it's plotted as lines.
        (variable.compress && should_compress(variable)) ? Int(requested) : 0
    else
        nothing
    end

    if !isnothing(precision)
        hit = get(cache, qualified, nothing)
        if !isnothing(hit) && hit[1] == precision
            return hit[2]
        end
    end

    new_data = if precision == METADATA_PRECISION
        ArrayMetadata(eltype(data), collect(size(data)))
    elseif !isnothing(precision)
        ws = get!(() -> ZfpWorkspace(), state.zfp_workspaces, qualified)
        compress_array(ws, data; precision)
    else
        data
    end

    view = if new_data === data
        variable
    else
        @set variable.data = new_data
    end

    if !isnothing(precision)
        cache[qualified] = (precision, view)
    end
    return view
end

# Build a TrainData payload for one client by composing per-(sub)variable views
# from the shared cache. Falls through to the parent's existing subvariables
# dict when nothing actually changed for any subvar (avoids an allocation).
function build_client_view!(state::EngineState, variable::VariableData,
                            subscriptions::Dict{String, Int},
                            cache::Dict{String, Tuple{Int, VariableData}})
    parent_view = client_view_for(state, variable, variable.name, subscriptions, cache)

    if isempty(variable.subvariables)
        return parent_view
    end

    new_subvars = Dict{String, Any}()
    any_changed = false
    for (qualified, subvar) in variable.subvariables
        sub_view = client_view_for(state, subvar, qualified, subscriptions, cache)
        any_changed |= sub_view !== subvar
        new_subvars[qualified] = sub_view
    end

    if !any_changed && parent_view === variable
        return variable
    end

    return @set parent_view.subvariables = new_subvars
end

function forward_output(state::EngineState, stream_output)
    cache = Dict{String, Tuple{Int, VariableData}}()
    for data in stream_output
        empty!(cache)
        for (id, client) in state.clients
            try
                view = build_client_view!(state, data, client.subscriptions, cache)
                Protocol.server_send(client.websocket, TrainData([view]))
            catch ex
                @warn "Couldn't forward data to client '$(id)'" exception=ex
            end
        end
    end
end

# Installed as the context's on_parameter_changed hook: broadcast an
# engine-initiated parameter change to every connected client so their GUIs
# stay in sync. Runs on proc 1 (where set_parameter calls the hook).
function broadcast_parameter_changed(state::EngineState, name, value)
    msg = Protocol.ParameterChanged(Context.Parameter(name, value))
    for (id, client) in state.clients
        try
            Protocol.server_send(client.websocket, msg)
        catch ex
            @warn "Failed to broadcast set_parameter to client '$(id)'" exception=ex
        end
    end
end

# Periodically broadcast a snapshot of every variable channel's drop count,
# fill level, and capacity to all connected clients. Used by the GUI to color
# pipeline edges by load.
function broadcast_channel_stats(state::EngineState; period=1.0)
    stats = Dict{Tuple{String, String}, Context.ChannelStat}()

    while !state.stop_event.set
        sleep(period)
        # Re-read state.ctx each iteration — LoadContext swaps it for a fresh
        # ContextState, so capturing it outside the loop would leave us pinned
        # to the never-running default context.
        ctx = state.ctx
        if !ctx.is_running[] || isempty(state.clients)
            continue
        end

        empty!(stats)
        try
            # Karabo edges are served by a two-stage channel chain
            # (input -> extractor -> consumer). Drops may land on either
            # stage, so sum both into a single edge entry keyed by
            # (dep_name, consumer); size reflects the downstream stage
            # since that's what the consumer actually sees queued.
            for (dep_name, downstream) in ctx.external_dependency_channels
                input_name = ctx.dep_to_input[dep_name]
                up = Context.channel_stat(ctx.input_variable_channels[input_name][dep_name])
                for (consumer, channel) in downstream
                    ds = Context.channel_stat(channel)
                    stats[(dep_name, consumer)] = Context.ChannelStat(up.drops + ds.drops, ds.size, ds.capacity)
                end
            end
            for (producer, downstream) in ctx.variable_channels
                for (consumer, channel) in downstream
                    stats[(producer, consumer)] = Context.channel_stat(channel)
                end
            end
        catch ex
            @warn "Failed to gather channel stats" exception=(ex, catch_backtrace())
            continue
        end

        msg = Protocol.PipelineStats(stats, copy(ctx.input_rates))
        for client in values(state.clients)
            try
                Protocol.server_send(client.websocket, msg)
            catch ex
                @debug "Failed to send PipelineStats to client" exception=(ex, catch_backtrace())
            end
        end
    end
end

"""Close connections to all clients and optionally the websocket listener."""
function shutdown(state::EngineState, ws_server=nothing)
    if ws_server != nothing
        close(ws_server)
    end

    for client in values(state.clients)
        close(client.websocket)
    end
end

function handle_message(msg::AbstractMessage, state::EngineState, id, request_id::Union{MessageId, Nothing})
    client_state = state.clients[id]
    ws = client_state.websocket
    reply_to = request_id

    if msg isa Ping
        client_state.last_heartbeat = time()
        Protocol.server_send(ws, Pong(); reply_to)
    elseif msg isa Shutdown
        @info "Received shutdown request from client $(id)"
        shutdown(state)
        notify(state.stop_event)

    elseif msg isa GetRoutingRules
        Protocol.server_send(ws, RoutingRules(state.routing_rules); reply_to)

    elseif msg isa GetRemapRules
        Protocol.server_send(ws, RemapRules(state.remap_rules); reply_to)

    elseif msg isa SetRoutingRules
        state.routing_rules = msg.rules
        write_routing_rules(msg.rules)
        @info "Updated routing rules" n=length(msg.rules) path=engine_settings_path()
        Protocol.server_send(ws, Ack(); reply_to)

        # Broadcast to all clients so concurrent editors stay in sync.
        broadcast = RoutingRules(state.routing_rules)
        for (other_id, other) in state.clients
            if other_id == id
                continue
            end
            try
                Protocol.server_send(other.websocket, broadcast)
            catch ex
                @warn "Failed to broadcast routing rules to client '$(other_id)'" exception=ex
            end
        end

    elseif msg isa GetDevices
        try
            devices = if isnothing(msg.topic)
                get_all_devices(state.webproxies)
            else
                Dict(msg.topic => get_devices(state.webproxies[msg.topic]))
            end
            Protocol.server_send(ws, Devices(devices); reply_to)
            @info "Responded to 'GetDevices' from $(id)"
        catch ex
            @error "Error in 'GetDevices', requested by $(id)" exception=(ex, catch_backtrace())
            Protocol.server_send(ws, Devices(Protocol.ExceptionMessage(ex, catch_backtrace())); reply_to)
        end

    elseif msg isa GetDeviceSchema
        schema = get_schema(KaraboDevice(msg.topic, msg.name))
        Protocol.server_send(ws, DeviceSchema(msg.topic, msg.name, schema); reply_to)
        @info "Responded to 'GetDeviceSchema' from $(id)"

    elseif msg isa GetDeviceProperty
        try
            wp = get_webproxy(KaraboDevice(msg.topic, msg.device))
            value = get_property(wp, msg.device, msg.property)
            Protocol.server_send(ws, DeviceProperty(msg.topic, msg.device, msg.property, value); reply_to)
            @info "Responded to 'GetDeviceProperty' ($(msg.device).$(msg.property)) from $(id)"
        catch ex
            @error "Error in 'GetDeviceProperty', requested by $(id)" exception=(ex, catch_backtrace())
            Protocol.server_send(ws, DeviceProperty(msg.topic, msg.device, msg.property, Protocol.ExceptionMessage(ex, catch_backtrace())); reply_to)
        end
    elseif msg isa GetEngineDir
        Protocol.server_send(ws, EngineDir(pkgdir(XfaEngine)); reply_to)

    elseif msg isa GetTrainmatchers
        trainmatchers = get_all_trainmatchers(state.webproxies)
        Protocol.server_send(ws, AvailableTrainmatchers(trainmatchers); reply_to)

    elseif msg isa LoadContext
        path = abspath(expanduser(msg.path))

        was_running = state.ctx.is_running[]
        if was_running
            Context.stop_pipeline(state.ctx)
        end

        new_ctx_or_ex = try
            dep_router = (topic, source) -> match_rule(state.routing_rules, topic, source)
            prelude = [:(using XfaEngine: KaraboInput)]
            ctx = Context.load_from_file(path; dep_router, prelude)
            ctx.forwarder = Base.Fix1(forward_output, state)
            ctx.on_parameter_changed = (name, value) -> broadcast_parameter_changed(state, name, value)
            ctx
        catch ex
            Protocol.ExceptionMessage(ex, catch_backtrace())
        end

        if new_ctx_or_ex isa ContextState
            state.ctx = new_ctx_or_ex
            if was_running
                @invokelatest Context.start_pipeline(state.ctx)
            end

            @info "Loaded context file $(path): $(state.ctx)"
            source = read(path, String)
            Protocol.server_send(ws, ContextInfo(state.ctx, source); reply_to)
        else
            @error "Loading context file at $(path) failed" exception=new_ctx_or_ex
            Protocol.server_send(ws, ContextInfo(new_ctx_or_ex, state.ctx.is_running[], ""); reply_to)
        end
    elseif msg isa ReviseCode
        @everywhere Revise.revise()
        @info "Revised source code"
        Protocol.server_send(ws, Ack(); reply_to)
    elseif msg isa ChangeParameter
        param = msg.parameter
        try
            Context.change_parameter(state.ctx, param)
            @info "ChangeParameter of $(param.name) to $(param.value)"
            Protocol.server_send(ws, Ack(); reply_to)

            # Broadcast the new value to all clients so they stay in sync and
            # the originating client can confirm the engine accepted the value.
            broadcast = ParameterChanged(param)
            for (other_id, other) in state.clients
                try
                    Protocol.server_send(other.websocket, broadcast)
                catch ex
                    @warn "Failed to broadcast ParameterChanged to client '$(other_id)'" exception=ex
                end
            end
        catch ex
            @error "Failed to change parameter '$(param.name)'" exception=(ex, catch_backtrace())
            Protocol.server_send(ws, Ack(Protocol.ExceptionMessage(ex, catch_backtrace())); reply_to)
        end
    elseif msg isa Start
        @info "Starting pipeline..."
        try
            Context.start_pipeline(state.ctx)
            Protocol.server_send(ws, Ack(); reply_to)
        catch ex
                @error "Failed to start pipeline" exception=(ex, catch_backtrace())
            Protocol.server_send(ws, Ack(Protocol.ExceptionMessage(ex, catch_backtrace())); reply_to)
        end
        @info "Started"
    elseif msg isa Stop
        @info "Stopping pipeline..."
        Context.stop_pipeline(state.ctx)
        Protocol.server_send(ws, Stopped(); reply_to)
        @info "Stopped"
    elseif msg isa SetVariableSubscriptions
        client_state.subscriptions = msg.variables
        Protocol.server_send(ws, Ack(); reply_to)
    elseif msg isa SetDebugMode
        @info "Setting debug mode: $(msg.enable)"
        if msg.enable
            ENV["JULIA_DEBUG"] = "XfaEngine"
        else
            delete!(ENV, "JULIA_DEBUG")
        end

        Protocol.server_send(ws, Ack(); reply_to)
    elseif msg isa SetRemoteRepl
        if msg.enable
            @info "Starting RemoteREPL"
            port, state.remoterepl_server = Sockets.listenany(27754)
            state.remoterepl_task = Threads.@spawn :samepool RemoteREPL.serve_repl(state.remoterepl_server)
            Protocol.server_send(ws, RemoteReplState(true, Int(port)); reply_to)
        else
            @info "Stopping RemoteREPL"
            close(state.remoterepl_server)
            wait(state.remoterepl_task)
            Protocol.server_send(ws, RemoteReplState(false, -1); reply_to)
        end
    else
        @error "Received unsupported message: $(typeof(msg))"
    end
end

"""Handle a single client."""
function handle_client(state::EngineState, id)
    client_state = state.clients[id]
    ws = client_state.websocket

    # Start by sending their identifier
    WebSockets.send(ws, id)
    @info "Connected to new client: $(id) 🙋"

    # If a context is already loaded, send it to the new client
    if !isempty(state.ctx.path)
        Protocol.server_send(ws, ContextInfo(state.ctx, read(state.ctx.path, String)))
    end

    try
        for msg_bytes in ws
            buffer = IOBuffer(msg_bytes)
            envelope = deserialize(buffer)::Envelope

            try
                @invokelatest handle_message(envelope.msg, state, id, envelope.id)
            catch ex
                @error "Caught exception when handling message from $(id)" exception=(ex, catch_backtrace())
            end
        end
    finally
        delete!(state.clients, id)
        @info "Disconnected from client $(id)"
    end
end

# Helper variables/functions to create amusing client IDs
const ID_PREFIXES = ["bothersome", "droopy", "deleterious", "morbid", "snobbish", "mirthful", "joyous", "blithe", "euphoric", "frolicsome"]
const ID_SUFFIXES = ["elf", "balrog", "wizard", "hobbit", "dwarf", "ent", "troll", "goblin", "tom", "dragon"]
create_id() = "$(rand(ID_PREFIXES))-$(rand(ID_SUFFIXES))"

function main(stop_event=Base.Event(); info_path=nothing, wait=true)
    websocket_port = getavailableport(1331)
    state = EngineState(; websocket_port, stop_event)
    global current_engine_state = state

    if haskey(ENV, "SASE")
        merge!(state.webproxies, Dict(instrument => WebProxy(address)
                                      for (instrument, address) in DEFAULT_WEBPROXY_ADDRESSES))
    elseif !endswith(gethostname(), ".desy.de")
        state.webproxies["localhost"] = WebProxy("localhost:8484")
    end

    loaded = load_routing_rules()
    if isnothing(loaded)
        # First run: seed one {topic, "*", first_matcher} rule per discovered
        # topic and persist, so subsequent runs read the file verbatim.
        rules = RoutingRule[]
        if !isempty(state.webproxies)
            try
                for (topic, matchers) in get_all_trainmatchers(state.webproxies)
                    if !isempty(matchers)
                        push!(rules, RoutingRule(topic, "*", first(matchers)[1]))
                    end
                end
            catch ex
                @warn "Failed to query trainmatchers while seeding routing rules" exception=(ex, catch_backtrace())
            end
        end
        write_routing_rules(rules)
        state.routing_rules = rules
        @info "Seeded routing rules" n=length(rules) path=engine_settings_path()
    else
        state.routing_rules = loaded
        @info "Loaded routing rules" n=length(loaded) path=engine_settings_path()
    end

    state.remap_rules = load_remap_rules()
    @info "Loaded remap rules" n=length(state.remap_rules) path=engine_settings_path()

    ws_server = WebSockets.listen!("0.0.0.0", state.websocket_port; maxframesize=Protocol.MAX_FRAME_SIZE) do ws
        id = create_id()
        try
            # Select their identifier
            while id in keys(state.clients)
                id = create_id()
            end

            # Create a client
            client = ClientState(; websocket=ws, last_heartbeat=time())
            state.clients[id] = client
            handle_client(state, id)
        catch ex
            if ex isa EOFError
                @info "Client $(id) disconnected"
            else
                @error "Error while handling client connecton" exception=(ex, catch_backtrace())
            end
        end
    end

    @info "Started listening on ws://$(gethostname()):$(websocket_port)"

    state.karabo_bridge_port = getavailableport(1332)

    # Write configuration
    worker_info = [(string(p), @fetchfrom p Dict("pid"      => getpid(),
                                                 "hostname" => gethostname()))
                   for p in procs()]
    worker_info = Dict(worker_info)
    worker_info["1"]["websocket-port"] = websocket_port
    worker_info["1"]["karabo-bridge-port"] = state.karabo_bridge_port

    if isnothing(info_path)
        info_path = abspath("worker-info.toml")
    else
        info_path = abspath(expanduser(info_path))
    end

    open(info_path, "w") do io
        TOML.print(io, worker_info)
    end

    @info "Wrote worker information to $(info_path)"

    state.channel_stats_task = Threads.@spawn :interactive broadcast_channel_stats(state)
    errormonitor(state.channel_stats_task)

    state.stop_task = Threads.@spawn :interactive try
        Base.wait(state.stop_event)
    catch ex
        if ex isa InterruptException
            @info "Shutdown requested, shutting down..."
        else
            @error "Exception thrown in main loop: $(ex)"
        end
    finally
        shutdown(state, ws_server)
    end

    if wait
        Base.wait(state.stop_task)
        Base.wait(state.channel_stats_task)
    end

    return state
end

end # module XfelAnalyserEngine
