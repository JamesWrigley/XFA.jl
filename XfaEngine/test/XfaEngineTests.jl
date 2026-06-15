module XfaEngineTests

__revise_mode__ = :eval

using Logging: Logging
using Sockets: Sockets, @ip_str, send, recv
using Statistics: mean
using Test: with_logger, TestLogger
using ReTest: @testset, @test, @test_throws, @test_logs


using ZMQ: ZMQ
using HTTP: HTTP, WebSockets
using JSON3: JSON3
using OrderedCollections: OrderedDict as OD
using DataStructures: CircularBuffer, capacity
using FHist: bincounts, bincenters, binedges
using DimensionalData: DimensionalData as DD, DimArray

using XfaEngine: XfaEngine, Context, KaraboBridge, Protocol, RoutingRule, match_rule,
    build_client_view!, is_scalar_data, ArrayMetadata, EngineState
using XfaEngine.ZfpWorkspaces: ZfpWorkspace, CompressedArray, compress_array,
    decompress_array, decompress_array!, allocate_array, restore_dims, should_compress
using XfaEngine: XfaEngine as engine
using XfaContext: @Variable, @karabo_str, VariableData, Dependency, DependencyKind,
    DepKind_Variable, DepKind_Subvariable, DepKind_Karabo, DepKind_Group, DepKind_GroupParameter,
    karabo_dependency, subvariable_dependency, group_dependency, group_parameter_dependency,
    XfaContextException, Parameter, FunctionArgument, KaraboDevice, CircularChannel, drop_count,
    PlotSpec, LayerSpec
using XfaEngine.KaraboBridge: KaraboBridgeClient, KaraboBridgeServer, ThreadsafeSocket


# Splices the engine-side KaraboInput group into a context module's namespace,
# mirroring what the engine passes to load_from_* at runtime.
const KARABO_PRELUDE = [:(using XfaEngine: KaraboInput)]

keyset(dict) = Set(keys(dict))

function test_connect(port=1331)
    client = Client()

    t = Threads.@spawn WebSockets.open(address) do ws
        client.websocket = ws

        id = WebSockets.receive(ws)
        client.client_id = id
        WebSockets.receive(ws) # engine directory

        for msg_bytes in ws
            buffer = IOBuffer(msg_bytes)
            msg::AbstractMessage = deserialize(buffer)
            @show msg
        end

        @info "Connection to $(address) closed"
    end

    return client, errormonitor(t)
end

function server_exists(port)
    try
        sock = Sockets.connect(port)
        close(sock)
        return true
    catch
        return false
    end
end

function mock_webproxy(f::Function, port, bridge_port=-1; slot_calls=nothing)
    server = HTTP.serve!(Sockets.localhost, port) do request
        if request.target == "/devices.json"
            return HTTP.Response(read(joinpath(@__DIR__, "mid-devices.json"), String))
        elseif endswith(request.target, "/slot/subscribeSources.json")
            if !isnothing(slot_calls)
                push!(slot_calls, JSON3.read(request.body, Dict{String, Any}))
            end
            # Mirrors the webproxy's SlotResponse format: the device reply is
            # nested under "reply" with {value, timestamp, tid}-wrapped leaves.
            return HTTP.Response("""{"success": true, "reason": "",
                                     "reply": {"success": {"value": true, "timestamp": 0, "tid": 0},
                                               "ttl": {"value": 0.5, "timestamp": 0, "tid": 0}}}""")
        elseif endswith(request.target, "/config.json")
            return HTTP.Response("""{"zmqOutputs": [{"address": "tcp://localhost:$(bridge_port)"}]}""")
        else
            return HTTP.Response(404, "Path not supported")
        end
    end

    try
        f()
    finally
        close(server)
    end
end

function temp_engine(f::Function; log=Logging.global_logger())
    mktemp() do info_path, io
        stop_event = Base.Event()
        state = with_logger(log) do
            XfaEngine.main(stop_event; info_path, wait=false)
        end
        port = state.websocket_port
        @test server_exists(port)

        address = "ws://localhost:$(port)"
        try
            f(address, stop_event, info_path)
        finally
            notify(state.stop_event)
            wait(state.stop_task)
        end
    end
end

@testset "Engine" begin
    # Smoke test
    event = Base.Event()
    mktemp() do info_path, io
        # Run the engine within a TestLogger so we don't see the logs
        log = TestLogger()
        t = Threads.@spawn with_logger(log) do
            XfaEngine.main(event; info_path)
        end

        @test timedwait(() -> isfile(info_path), 10) == :ok

        notify(event)
        @test timedwait(() -> istaskdone(t), 10) == :ok

        @test occursin("[1]", read(info_path, String))
    end

    log = TestLogger()
    temp_engine(; log) do address, stop_event, info_path
        WebSockets.open(address) do ws
            # Test that we get a valid ID (the only thing sent
            # unsolicited on connect)
            id = WebSockets.receive(ws)
            @test id isa String
            @test length(id) > 5

            # Engine directory and trainmatchers are now only sent on request
            Protocol.client_send(ws, Protocol.GetEngineDir())
            engine_dir_msg = Protocol.receive(ws).msg
            @test engine_dir_msg isa Protocol.EngineDir
            @test engine_dir_msg.path == pkgdir(XfaEngine)

            Protocol.client_send(ws, Protocol.GetTrainmatchers())
            @test Protocol.receive(ws).msg isa Protocol.AvailableTrainmatchers

            # Test Ping
            Protocol.client_send(ws, Protocol.Ping())
            @test Protocol.receive(ws).msg isa Protocol.Pong

            # Test GetDevices
            webproxy_port = XfaEngine.getavailableport(8484)
            mock_webproxy(webproxy_port) do
                Protocol.client_send(ws, Protocol.GetDevices())
                @test Protocol.receive(ws).msg isa Protocol.Devices
            end

            # Test LoadContext
            mktemp() do path, io
                # Test loading an invalid context
                # write(path, "@Variable x -> foo")
                # Protocol.client_send(ws, Protocol.LoadContext(path))
                # msg = Protocol.receive(ws).msg
                # @test msg isa Protocol.ContextInfo
                # @test msg.info isa Exception

                # Test loading a valid context
                write(path, """
                            p = Parameter(0)
                            @Variable x -> karabo"foo.bar"
                            """)
                Protocol.client_send(ws, Protocol.LoadContext(path))
                msg = Protocol.receive(ws).msg
                @test msg isa Protocol.ContextInfo
                @test msg.info isa Dict
                @test haskey(msg.info["dag"], "x")
            end

            # Test ChangeParameter — engine should ack and then broadcast the
            # new value back so all clients can sync.
            Protocol.client_send(ws, Protocol.ChangeParameter(Parameter("p", 1)))
            ack = Protocol.receive(ws).msg
            @test ack isa Protocol.Ack
            @test isnothing(ack.error)
            changed = Protocol.receive(ws).msg
            @test changed isa Protocol.ParameterChanged
            @test changed.parameter.name == "p"
            @test changed.parameter.value == 1

            # A failing update_handler must surface as Ack(error) and not emit
            # a ParameterChanged broadcast.
            mktemp() do path, io
                write(path, """
                            p = Parameter(0) do v
                                error("boom")
                            end
                            @Variable x -> karabo"foo.bar"
                            """)
                Protocol.client_send(ws, Protocol.LoadContext(path))
                msg = nothing
                while !(msg isa Protocol.ContextInfo); msg = Protocol.receive(ws).msg end
                @test msg.info isa Dict
            end
            Protocol.client_send(ws, Protocol.ChangeParameter(Parameter("p", 2)))
            ack = Protocol.receive(ws).msg
            @test ack isa Protocol.Ack
            @test !isnothing(ack.error)
            @test occursin("boom", ack.error.text)

            # Test ReviseCode
            Protocol.client_send(ws, Protocol.ReviseCode())
            @test Protocol.receive(ws).msg isa Protocol.Ack
        end
    end

    change_param_logs = [x.message for x in log.logs if occursin("ChangeParameter of p", x.message)]
    @test length(change_param_logs) == 1

    # launcher_script = joinpath(dirname(dirname(@__FILE__)), "src/launcher.jl")
    # executable = Base.julia_cmd()
    # environment = Base.active_project()

    # mktempdir() do tmpdir
    #     cd(tmpdir) do
    #         run(`$(executable) --project=$(environment) --startup-file=no --color=no $(launcher_script)`)
    #     end
    # end


    @testset "Channel stats" begin
        # End-to-end check that the engine periodically pushes a PipelineStats
        # message summarising drops/size/capacity for each variable channel.
        # A slow downstream variable guarantees drops accumulate.
        log = TestLogger()
        temp_engine(; log) do address, stop_event, info_path
            WebSockets.open(address) do ws
                WebSockets.receive(ws) # client id

                # Load a slow-consumer pipeline
                mktemp() do path, io
                    write(path, """
                    @Input function input(::Context.MockInput, output)
                        for tid in 1:1000
                            put!(output, (tid, Dict("motor" => Dict("pos" => tid))))
                        end
                    end
                    x = Context.MockInput()

                    @Variable function slow(data -> karabo"motor.pos")
                        sleep(0.01)
                        return data
                    end
                    """)
                    Protocol.client_send(ws, Protocol.LoadContext(path))
                    while !(Protocol.receive(ws).msg isa Protocol.ContextInfo) end
                end

                Protocol.client_send(ws, Protocol.Start())
                while !(Protocol.receive(ws).msg isa Protocol.Ack) end

                # Collect messages until we get a PipelineStats with a non-zero
                # drop count on the (motor.pos, slow) channel, or time out.
                key = ("motor.pos", "slow")
                stats = nothing
                deadline = time() + 10.0
                while isnothing(stats) || (time() < deadline && stats.drops == 0)
                    msg = Protocol.receive(ws).msg
                    if msg isa Protocol.PipelineStats && msg.channel_stats[key].drops > 0
                        stats = msg.channel_stats[key]
                    end
                end

                @test stats.drops > 0
                @test stats.capacity == 100
                @test 0 <= stats.size <= 100
            end
        end
    end
end

@testset "Message tracking" begin
    log = TestLogger()
    temp_engine(; log) do address, stop_event, info_path
        WebSockets.open(address) do ws
            # Consume the client ID
            WebSockets.receive(ws)

            # Test that send always assigns an ID and the server
            # echoes it back as reply_to
            id = Protocol.client_send(ws, Protocol.Ping())
            @test id > 0
            envelope = Protocol.receive(ws)
            @test envelope isa Protocol.Envelope
            @test envelope.id < 0
            @test envelope.reply_to == id
            @test envelope.msg isa Protocol.Pong

            # Test that Ack messages carry reply_to for fire-and-forget
            # messages
            id1 = Protocol.client_send(ws, Protocol.SetRoutingRules(RoutingRule[]))
            id2 = Protocol.client_send(ws, Protocol.SetRoutingRules(RoutingRule[]))
            env1 = Protocol.receive(ws)
            env2 = Protocol.receive(ws)
            @test env1.msg isa Protocol.Ack
            @test env2.msg isa Protocol.Ack
            @test Set([env1.reply_to, env2.reply_to]) == Set([id1, id2])
        end
    end
end

function getavailableport(port_hint; interface=ip"127.0.0.1")
    port_range_end = min(65535, port_hint + 100)
    available_port = -1

    for port in port_hint:port_range_end
        try
            s = Sockets.listen(interface, port)
            close(s)
            return port
        catch ex
            continue
        end
    end

    error("Could not find an available port between $(port_hint) and $(port_range_end)")
end

@testset "ThreadsafeSocket" begin
    s1 = ZMQ.Socket(ZMQ.PUSH)
    s2 = ZMQ.Socket(ZMQ.PULL)

    try
        ZMQ.bind(s1, "tcp://*:5555")
        ZMQ.connect(s2, "tcp://localhost:5555")

        ts1 = ThreadsafeSocket(s1)
        ts2 = ThreadsafeSocket(s2)
        ts1.sndhwm = 100

        # Smoke test
        send(ts1, "foo")
        @test recv(ts2, String) == "foo"

        # Multi-threaded test. Spawn many tasks simultaneously reading and
        # writing to the sockets.
        n_msgs = s1.sndhwm ÷ 2
        msgs = Channel{Int}(n_msgs)
        for i in 1:n_msgs
            Threads.@spawn send(ts1, i)
        end
        @sync for i in 1:n_msgs
            Threads.@spawn put!(msgs, recv(ts2, Int))
        end
        close(msgs)
        msgs = collect(msgs)

        @test sort(msgs) == 1:n_msgs

        @test isopen(ts1)
        close(ts1)
        @test !isopen(ts1)
        @test istaskdone(ts1.handler)
    finally
        close(s1)
        close(s2)
    end
end

function karabo_bridge_test_state(f::Function, endpoint)
    server = KaraboBridgeServer(endpoint)
    client = KaraboBridgeClient(endpoint)

    try
        f(client, server)
    finally
        close(client)
        close(server)
    end
end

@testset "Karabo bridge" begin
    # Create server and client
    port = getavailableport(42000)
    endpoint = "tcp://127.0.0.1:$(port)"

    @testset "Basic tests" begin
        karabo_bridge_test_state(endpoint) do client, server
            # Start the server
            KaraboBridge.startbridge(server)
            @test isopen(server.channel)

            # The server should now be bound to the port
            @test_throws Base.IOError Sockets.listen(ip"127.0.0.1", port)

            # Trying to start it twice should fail
            @test_throws ErrorException KaraboBridge.startbridge(server)

            # Stop the server
            close(server)
            @test timedwait(() -> !server.is_running, 5) == :ok
        end

        karabo_bridge_test_state(endpoint) do client, server
            # Create some test data
            dummy_data = Dict("foo" => Dict(
                "string" => "hello world!",
                "scalar" => 42.314,
                "boolean" => true,
                "list" => ["foo", "bar", 42, 3.14],
            ))
            for type in [Bool,
                         Float16, Float32, Float64,
                         Int8, Int16, Int32, Int64,
                         UInt8, UInt16, UInt32, UInt64]
                # These arrays should use zero-copy transfer
                dummy_data["foo"]["big_$(lowercase(string(type)))_array"] = rand(type, 1000, 1000)
                # These arrays should be serialized, except for Float16 since MsgPack
                # doesn't support Float16.
                dummy_data["foo"]["small_$(lowercase(string(type)))_array"] = rand(type, 10)
            end

            # Send the test data and ensure it's received by the client
            KaraboBridge.startbridge(server)
            put!(server, dummy_data)
            data, metadata = take!(client)
            @test dummy_data == data
        end
    end

    @testset "BufferPool" begin
        karabo_bridge_test_state(endpoint) do client, server
            KaraboBridge.startbridge(server)
            pool = KaraboBridge.BufferPool()

            send_payload = Dict("src" => Dict("arr" => UInt16[1, 2, 3, 4, 5]))
            put!(server, send_payload)
            data, _ = take!(client, pool)
            @test data == send_payload

            # Same (source, path) should reuse the same underlying Vector
            # across rotations within VARIABLE_CHANNEL_SIZE trains.
            ring = pool[("src", "arr")]
            buf_first_round = ring.buffers[1]

            for _ in 1:Context.VARIABLE_CHANNEL_SIZE
                put!(server, send_payload)
                take!(client, pool)
            end
            @test ring.buffers[1] === buf_first_round

            # A different (source, path) gets its own ring.
            other = Dict("other" => Dict("v" => Float32[1.0, 2.0]))
            put!(server, other)
            data, _ = take!(client, pool)
            @test data == other
            @test haskey(pool, ("other", "v"))
            @test pool[("other", "v")] isa KaraboBridge.BufferRing{Float32}
        end
    end
end

# Test postprocessors, also in Main for load_from_string access.
@eval Main module PostprocessorLibrary
    using Statistics: mean
    import XfaContext as Context
    using XfaContext: AbstractPostprocessor, Parameter

    struct TestMean <: AbstractPostprocessor end
    Context.default_name(::TestMean) = "mean"
    (::TestMean)(data) = mean(data)

    mutable struct TestWindow <: AbstractPostprocessor
        size::Parameter{Int}
    end
    TestWindow(; size=10) = TestWindow(Parameter(size))
    Context.default_name(::TestWindow) = "window"
    (w::TestWindow)(data) = data[1:min(end, w.size[])]
end

@testset "KaraboInput" begin
    # Instantiating the engine-side KaraboInput plugin registers its input
    # under <instance>.stream, backed by XfaEngine.stream.
    ctx = Context.load_from_string("""
    bridge = KaraboInput(; trainmatcher=KaraboDevice("MATCHER"))
    """; prelude=KARABO_PRELUDE)
    @test haskey(ctx.inputs, "bridge.stream")
    @test ctx.functions["bridge.stream"] === XfaEngine.stream

    @testset "Source leases" begin
        # In automatic configuration mode the bridge should lease its
        # dependencies' sources through the subscribeSources slot and keep
        # renewing them while streaming.
        webproxy_port = XfaEngine.getavailableport(8485)
        bridge_port = XfaEngine.getavailableport(42000)
        bridge_server = KaraboBridgeServer("tcp://localhost:$(bridge_port)")
        KaraboBridge.startbridge(bridge_server)

        ctx = Context.load_from_string("""
        bridge = KaraboInput(; trainmatcher=KaraboDevice("localhost//MATCHER"))
        bridge._mock_sources = String[]

        @Variable foo -> karabo"foo.x"
        """; prelude=KARABO_PRELUDE)

        webproxies = Dict("localhost" => XfaEngine.WebProxy("localhost:$(webproxy_port)"))
        XfaEngine.current_engine_state = XfaEngine.EngineState(; webproxies)

        slot_calls = []
        put!(bridge_server, Dict("foo" => Dict("x" => 42.0)))
        mock_webproxy(webproxy_port, bridge_port; slot_calls) do
            Context.run(ctx) do
                @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
                @test take!(ctx.stream_output) == VariableData(0, "foo", 42.0)

                # The initial subscription leased the karabo dependency, and
                # the bridge picked up the mocked TTL from the reply
                @test slot_calls[1]["sources"] == ["foo.x"]
                bridge = ctx.groups["bridge"]
                @test bridge.lease_ttl == 0.5

                # The lease is renewed periodically (every TTL/2 = 0.25s)
                n = length(slot_calls)
                @test timedwait(() -> length(slot_calls) > n, 5) == :ok
            end
        end
        close(bridge_server)
    end
end

@testset "Scheduler" begin
    @testset "Routing" begin
        @testset "match_rule" begin
            # Empty rules always miss; literal and glob patterns both work;
            # first match wins when multiple rules could apply.
            @test isnothing(match_rule(RoutingRule[], "T", "foo"))

            rules = [RoutingRule("T1", "exact", "DEV_A"),
                     RoutingRule("T1", "foo.*", "DEV_B"),
                     RoutingRule("*", "*", "DEV_FALLBACK")]
            @test match_rule(rules, "T1", "exact") == "DEV_A"
            @test match_rule(rules, "T1", "foo.bar") == "DEV_B"
            @test match_rule(rules, "T2", "anything") == "DEV_FALLBACK"

            # More-specific rule only wins if it's ordered first
            reversed = [RoutingRule("*", "*", "DEV_FALLBACK"),
                        RoutingRule("T1", "exact", "DEV_A")]
            @test match_rule(reversed, "T1", "exact") == "DEV_FALLBACK"

            # Character-class and ?-wildcard globs
            class_rules = [RoutingRule("*", "cam[0-9]", "DEV_CAM"),
                           RoutingRule("*", "mot?r", "DEV_MOTOR")]
            @test match_rule(class_rules, "T", "cam3") == "DEV_CAM"
            @test match_rule(class_rules, "T", "motor") == "DEV_MOTOR"
            @test isnothing(match_rule(class_rules, "T", "camera"))
        end

        @testset "build_dep_routing with rules" begin
            # Two bridges, different trainmatcher devices. Rules are matched
            # against the karabo dependency's source/device name (e.g. for
            # karabo"foo.bar" the source is "foo", not "foo.bar").
            ctx_src = """
            bridge_a = KaraboInput(; trainmatcher=KaraboDevice("T1//DEV_A"))
            bridge_a._mock_sources = String[]

            bridge_b = KaraboInput(; trainmatcher=KaraboDevice("T2//DEV_B"))
            bridge_b._mock_sources = String[]

            @Variable foo -> karabo"foo.bar"
            @Variable special -> karabo"T1//special.src"
            """

            # No rules: topic-match routes the prefixed dep; unprefixed dep has no
            # topic/source match and two inputs exist, so it errors.
            @test_throws XfaContextException Context.load_from_string(ctx_src; prelude=KARABO_PRELUDE)

            # Rule forces source "foo" to bridge_b (device name DEV_B) regardless
            # of topic. The topicked dep falls through to the topic-match heuristic.
            rules = [RoutingRule("*", "foo", "DEV_B")]
            ctx = Context.load_from_string(ctx_src; dep_router=(t, s) -> match_rule(rules, t, s), prelude=KARABO_PRELUDE)
            @test ctx.dep_to_input["foo.bar"] == "bridge_b.stream"
            @test ctx.dep_to_input["T1//special.src"] == "bridge_a.stream"

            # Rule pointing at a device that isn't among the inputs falls through
            # to the existing heuristics (the trailing rule keeps foo routable).
            rules = [RoutingRule("*", "special", "NONEXISTENT_DEV"),
                     RoutingRule("*", "*", "DEV_A")]
            ctx = Context.load_from_string(ctx_src; dep_router=(t, s) -> match_rule(rules, t, s), prelude=KARABO_PRELUDE)
            @test ctx.dep_to_input["T1//special.src"] == "bridge_a.stream"

            # First-match-wins: a specific rule overrides the catch-all below it.
            rules = [RoutingRule("*", "foo", "DEV_A"),
                     RoutingRule("*", "*", "DEV_B")]
            ctx = Context.load_from_string(ctx_src; dep_router=(t, s) -> match_rule(rules, t, s), prelude=KARABO_PRELUDE)
            @test ctx.dep_to_input["foo.bar"] == "bridge_a.stream"
            @test ctx.dep_to_input["T1//special.src"] == "bridge_b.stream"

            # Topic-qualified input ("T//DEV") disambiguates when multiple
            # topics have devices with the same name.
            same_name_src = raw"""
            bridge_a = KaraboInput(; trainmatcher=KaraboDevice("T1//DEV"))
            bridge_a._mock_sources = String[]

            bridge_b = KaraboInput(; trainmatcher=KaraboDevice("T2//DEV"))
            bridge_b._mock_sources = String[]

            @Variable foo -> karabo"foo.bar"
            """
            rules = [RoutingRule("*", "foo", "T2//DEV")]
            ctx = Context.load_from_string(same_name_src; dep_router=(t, s) -> match_rule(rules, t, s), prelude=KARABO_PRELUDE)
            @test ctx.dep_to_input["foo.bar"] == "bridge_b.stream"

            rules = [RoutingRule("*", "foo", "T1//DEV")]
            ctx = Context.load_from_string(same_name_src; dep_router=(t, s) -> match_rule(rules, t, s), prelude=KARABO_PRELUDE)
            @test ctx.dep_to_input["foo.bar"] == "bridge_a.stream"
        end
    end
end

@testset "Context builtins" begin
    @testset "KaraboBridge" begin
        port = getavailableport(42000)
        address = "tcp://localhost:$(port)"
        bridge_server = KaraboBridgeServer(address)
        KaraboBridge.startbridge(bridge_server)

        ctx = Context.load_from_string("""
        bridge = KaraboInput(; trainmatcher=KaraboDevice("MATCHER"), sources=["foo.x"])
        bridge._mock_sources = String[]
        bridge.manual_configuration[] = true
        bridge.address[] = "$(address)"

        @Variable foo -> karabo"foo.x"
        """; prelude=KARABO_PRELUDE)

        # Make a mock engine so we can use the mock webproxy
        webproxies = Dict("localhost" => XfaEngine.WebProxy("localhost:8484"))
        XfaEngine.current_engine_state = XfaEngine.EngineState(; webproxies)

        # Simple example with two trains of data
        put!(bridge_server, Dict("foo" => Dict("x" => 42.0)))
        put!(bridge_server, Dict("foo" => Dict("x" => 40.0)))
        mock_webproxy(8484, port) do
            Context.run(ctx) do
                @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
                @test take!(ctx.stream_output) == VariableData(0, "foo", 42.0)

                @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
                @test take!(ctx.stream_output) == VariableData(1, "foo", 40.0)
            end
        end
        close(bridge_server)

        # Stopping the pipeline again shouldn't do anything
        Context.stop_pipeline(ctx)

        # We should be able to start the same context again
        bridge_server = KaraboBridgeServer("tcp://localhost:$(port)")
        KaraboBridge.startbridge(bridge_server)
        put!(bridge_server, Dict("foo" => Dict("x" => 38.0)))
        mock_webproxy(8484, port) do
            Context.run(ctx) do
                @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
                @test take!(ctx.stream_output) == VariableData(0, "foo", 38.0)
            end
        end

        close(bridge_server)
    end
end

@testset "Subscription filtering" begin
    @test is_scalar_data(1.0)
    @test is_scalar_data("foo")
    @test is_scalar_data(fill(1.0))
    @test !is_scalar_data([1, 2, 3])

    state = EngineState()
    cache() = Dict{String, Tuple{Int, VariableData}}()
    sub(pairs::Pair{String, Int}...) = Dict{String, Int}(pairs...)

    # Scalars always pass through. Non-compressible arrays (Int, length below
    # the compression threshold) round-trip raw when subscribed, and become
    # ArrayMetadata when not.
    scalar = VariableData(0, "s", 42)
    array = VariableData(0, "a", [1, 2, 3])
    @test build_client_view!(state, scalar, sub(), cache()) === scalar
    f = build_client_view!(state, array, sub(), cache())
    @test f.data isa ArrayMetadata
    @test f.data.eltype === Int
    @test f.data.size == [3]
    @test build_client_view!(state, array, sub("a" => -1), cache()) === array

    # Subvariables follow the same rule under their qualified name. The
    # subvariables dict is keyed by the qualified name (as produced by
    # @add_subvariable), and subscriptions must look it up under that same key.
    bar = VariableData(; tid=0, name="p", data=[1, 2],
                          subvariables=Dict{String, Any}(
                              "p.scalar" => VariableData(0, "p.scalar", 1.5),
                              "p.arr" => VariableData(0, "p.arr", [4, 5])))
    f = build_client_view!(state, bar, sub(), cache())
    @test f.data isa ArrayMetadata
    @test keyset(f.subvariables) == Set(["p.scalar", "p.arr"])
    @test f.subvariables["p.scalar"].data == 1.5
    @test f.subvariables["p.arr"].data isa ArrayMetadata

    # Subscribing to the qualified subvariable name delivers the real array.
    f = build_client_view!(state, bar, sub("p.arr" => -1), cache())
    @test f.data isa ArrayMetadata
    @test f.subvariables["p.arr"].data == [4, 5]

    # Subscribing to the bar does not implicitly subscribe its subvariables.
    f = build_client_view!(state, bar, sub("p" => -1), cache())
    @test f.data == [1, 2]
    @test f.subvariables["p.arr"].data isa ArrayMetadata

    # Re-prepending the bar name (the historical bug) would look up
    # "p.p.arr" and miss the subscription — make sure that doesn't happen.
    f = build_client_view!(state, bar, sub("p.p.arr" => -1), cache())
    @test f.subvariables["p.arr"].data isa ArrayMetadata

    # Compressible payload: a long enough Float array triggers ZFP. With two
    # clients sharing the same precision the cache reuses the compressed view.
    big = VariableData(; tid=0, name="big", data=randn(Float64, 600))
    c = cache()
    a = build_client_view!(state, big, sub("big" => -1), c)
    b = build_client_view!(state, big, sub("big" => -1), c)
    @test a.data isa CompressedArray
    @test a === b
    # A different precision recompresses and overwrites the cache slot.
    d = build_client_view!(state, big, sub("big" => 8), c)
    @test d.data isa CompressedArray
    @test d !== a
    @test c["big"][1] == 8
end

@testset "Serialization" begin
    ctx = Context.load_from_string(raw"""
        using Main.PostprocessorLibrary: TestWindow

        bridge = KaraboInput(; trainmatcher=KaraboDevice(""))
        bridge._mock_sources = String[]

        period = Parameter(2π)
        roi = Parameter(Context.RectROI())

        @Variable xgm -> karabo"xgm.intensity"

        @Variable function foo() 42 end

        @Variable function bar(data -> xgm)
            @add_subvariable("max_data", max(data))
            @postprocess(TestWindow(; size=5))
            @display roi
            mean(data)
        end
        """; prelude=KARABO_PRELUDE)

    @test Context.to_dict(ctx) == Dict("inputs" => Dict("bridge.stream" => ["bridge"]),
                                       "groups" => ["bridge"],
                                       "dag" =>          Dict("xgm" => OD("data" => karabo"xgm.intensity"),
                                                              "foo" => OD(),
                                                              "bar" => OD("data" => Dependency("xgm"))),
                                       "subvariables" => Dict("xgm" => [],
                                                              "foo" => [],
                                                              "bar" => ["bar.max_data", "bar.window"]),
                                       "postprocessors" => Dict("bar" => ["bar.window"]),
                                       "postprocessor_origins" => Dict("bar.window" => "Main.PostprocessorLibrary.TestWindow"),
                                       "displays" => Dict("bar" => ["roi"]),
                                       "origins" => Dict("xgm" => "xgm",
                                                         "foo" => "foo",
                                                         "bar" => "bar",
                                                         "bridge" => "XfaEngine.KaraboInput",
                                                         "bridge.stream" => "XfaEngine.stream"),
                                       "parameters" => Dict("period" => Parameter("period", 2π),
                                                            "roi" => Parameter("roi", Context.RectROI()),
                                                            "bar.window.size" => Parameter("bar.window.size", 5),
                                                            "bridge.address" => Parameter("bridge.address", ""),
                                                            "bridge.trainmatcher" => Parameter("bridge.trainmatcher", KaraboDevice("", "")),
                                                            "bridge.manual_configuration" => Parameter("bridge.manual_configuration", false)),
                                       "dep_to_input" => Dict("xgm.intensity" => "bridge.stream"),
                                       "group_parameter_args" => Dict(),
                                       "path" => "")

    # Group variables that reference a group Parameter field record the
    # arg_name -> field mapping so the client can rewrite the right kwarg.
    ctx = Context.load_from_string(raw"""
        @Group mutable struct Foo
            source::Parameter{Dependency}
        end

        @Variable function foo(::Foo, data -> Foo.source)
            data
        end

        foo_group = Foo(; source=karabo"motor1.pos")
        """)
    @test Context.to_dict(ctx)["group_parameter_args"] ==
        Dict("foo_group.foo" => Dict("data" => "source"))

    # Subvariables of a grouped variable must be reported under the
    # group-qualified DAG name, not the bare function name. The client uses
    # these strings to build output-pin IDs; if they're not remapped the
    # downstream link's start_id won't match any pin.
    ctx = Context.load_from_string(raw"""
        @Group struct G end

        @Variable function gv(::G)
            @add_subvariable("sub", 1)
            0
        end

        g = G()
        """)
    @test Context.to_dict(ctx)["subvariables"]["g.gv"] == ["g.gv.sub"]
end

@testset "ZfpWorkspace" begin
    ws = ZfpWorkspace()

    @testset "should_compress" begin
        @test should_compress(zeros(600))
        @test should_compress(rand(UInt8, 500))
        @test !should_compress(zeros(100))
        @test !should_compress("string")
        @test !should_compress(zeros(Bool, 600))

        # A 2D array plotted as a bunch of lines (a color-bound layer) skips
        # compression; the same array without that layer still compresses, and
        # the line exception only applies to matrices.
        lines = [PlotSpec("p", [LayerSpec(; data="v", color=:pulseId)])]
        plain = [PlotSpec("p", [LayerSpec(; data="v")])]
        @test !should_compress(VariableData(; data=rand(100, 5), plot_specs=lines))
        @test should_compress(VariableData(; data=rand(100, 5), plot_specs=plain))
        @test should_compress(VariableData(; data=rand(600), plot_specs=lines))
    end

    # High precision pins the round-trip fidelity regardless of whatever
    # lossy default the engine currently uses.
    @testset "Float round-trip (all finite)" begin
        for T in (Float32, Float64), shape in ((1000,), (40, 40))
            arr = randn(T, shape)
            ca = compress_array(ws, arr; precision=15)
            @test !ca.promoted && isnothing(ca.nonfinite_mask)
            @test ca.original_eltype === T && Tuple(ca.shape) == shape
            out = decompress_array(ws, ca)
            @test eltype(out) === T && size(out) == shape
            @test maximum(abs, arr - out) < 1e-2
        end
    end

    @testset "Float round-trip with non-finites" begin
        a = rand(Float32, 2000)
        a[10] = NaN32
        a[100] = Inf32
        a[200] = -Inf32
        a[1500] = NaN32

        ca = compress_array(ws, a; precision=15)
        @test !isnothing(ca.nonfinite_mask)
        out = decompress_array(ws, ca)
        @test isnan(out[10]) && out[100] == Inf32 && out[200] == -Inf32 && isnan(out[1500])
        fin = isfinite.(a)
        @test maximum(abs, a[fin] - out[fin]) < 1e-2
    end

    # Int round-trip uses precision=0 (lossless) to exercise the
    # promote/demote machinery; the default lossy precision=15 would zero
    # out small integer values and obscure whether promotion is correct.
    @testset "Low-bit int promote/demote" begin
        for T in (Int8, UInt8, Int16, UInt16)
            arr = T.(rand(0:50, 800))
            ca = compress_array(ws, arr; precision=0)
            @test ca.promoted && ca.original_eltype === T
            out = decompress_array(ws, ca)
            @test eltype(out) === T && out == arr
        end
    end

    @testset "Native int (no promotion)" begin
        arr = Int32.(rand(-100:100, 1000))
        ca = compress_array(ws, arr; precision=0)
        @test !ca.promoted && !ca.clamped
        @test decompress_array(ws, ca) == arr
    end

    @testset "Native int out-of-range gets clamped" begin
        mag = Int32(2)^30 - one(Int32)
        arr = Int32[0, 1, -2, typemax(Int32), typemin(Int32), 100]
        ca = compress_array(ws, arr; precision=0)
        @test ca.clamped
        out = decompress_array(ws, ca)
        @test out == Int32[0, 1, -2, mag, -mag, 100]
    end

    @testset "decompress_array! into provided buffer" begin
        arr = randn(Float64, 800)
        ca = compress_array(ws, arr; precision=15)
        out = allocate_array(ca)
        @test eltype(out) === Float64 && size(out) == size(arr)
        decompress_array!(ws, out, ca)
        @test maximum(abs, arr - out) < 1e-2

        @test_throws ArgumentError decompress_array!(ws, zeros(Float32, 800), ca)
        @test_throws DimensionMismatch decompress_array!(ws, zeros(801), ca)
    end

    # DimArrays compress their parent array and ship the dimension info so the
    # client can rebuild the DimArray after decompression.
    @testset "DimArray round-trip" begin
        parent_data = randn(Float64, 30, 40)
        da = DimArray(parent_data, (DD.Y(1:30), DD.X(101:140));
                      name="image", metadata=Dict(:units => "eV"))

        @test should_compress(da)
        ca = compress_array(ws, da; precision=15)
        @test !isnothing(ca.dims)
        @test ca.dims.dim_names == [:Y, :X]
        @test ca.dims.name == "image" && ca.dims.metadata[:units] == "eV"

        out = decompress_array(ws, ca)
        @test out isa DD.DimArray
        @test DD.name(out) == "image" && DD.metadata(out)[:units] == "eV"
        @test DD.dims(out) == DD.dims(da)
        @test maximum(abs, parent(da) - parent(out)) < 1e-2

        # No-dims CompressedArrays pass through restore_dims unchanged.
        plain = compress_array(ws, randn(600))
        @test restore_dims(allocate_array(plain), plain) isa Vector{Float64}
    end
end

end
