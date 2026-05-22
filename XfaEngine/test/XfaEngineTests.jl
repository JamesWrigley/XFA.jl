module XfaEngineTests

__revise_mode__ = :eval

using Logging: Logging
using Sockets: Sockets, @ip_str, send, recv
using Statistics: mean
using Test: with_logger, TestLogger
using ReTest: @testset, @test, @test_throws, @test_logs

using ZMQ: ZMQ
using HTTP: HTTP, WebSockets
using OrderedCollections: OrderedDict as OD
using DataStructures: CircularBuffer, capacity
using FHist: bincounts, bincenters, binedges

using XfaEngine: XfaEngine, Context, KaraboBridge, Protocol, RoutingRule, match_rule,
    build_client_view!, is_scalar_data, ArrayMetadata, EngineState
using XfaEngine.ZfpWorkspaces: ZfpWorkspace, CompressedArray, compress_array,
    decompress_array, decompress_array!, allocate_array, should_compress
using XfaEngine.Context: @Variable, @karabo_str, VariableData, Dependency, DependencyKind,
    DepKind_Variable, DepKind_Subvariable, DepKind_Karabo, DepKind_Group, DepKind_GroupParameter,
    karabo_dependency, subvariable_dependency, group_dependency, group_parameter_dependency,
    XfaContextException, Parameter, FunctionArgument, KaraboDevice, CircularChannel, drop_count
using XfaEngine.KaraboBridge: KaraboBridgeClient, KaraboBridgeServer, ThreadsafeSocket


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

function mock_webproxy(f::Function, port, bridge_port=-1)
    server = HTTP.serve!(Sockets.localhost, port) do request
        if request.target == "/devices.json"
            return HTTP.Response(read(joinpath(@__DIR__, "mid-devices.json"), String))
        elseif endswith(request.target, "/set_sources.json")
            return HTTP.Response("""{"status": "ok"}""")
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

@testset "CircularChannel" begin
    # Basic FIFO behaviour when within capacity
    c = CircularChannel{Int}(3)
    @test isopen(c) && !isready(c)
    put!(c, 1)
    put!(c, 2)
    @test isready(c) && drop_count(c) == 0
    @test take!(c) == 1 && take!(c) == 2
    @test !isready(c)

    # Overwrite-oldest when full: 5 puts into capacity 3 → drops=2, remaining 3,4,5
    for i in 1:5
        put!(c, i)
    end
    @test drop_count(c) == 2
    @test [take!(c) for _ in 1:3] == [3, 4, 5]

    # take! blocks until put! and is woken by notify.
    c = CircularChannel{Int}(2)
    t = Threads.@spawn take!(c)
    @test timedwait(() -> istaskstarted(t), 10) == :ok
    put!(c, 42)
    @test fetch(t) == 42

    # close() drains remaining items then errors; put! on closed also errors.
    c = CircularChannel{Int}(2)
    put!(c, 7)
    close(c)
    @test !isopen(c)
    @test take!(c) == 7
    @test_throws InvalidStateException take!(c)
    @test_throws InvalidStateException put!(c, 1)

    # close() wakes blocked waiters with InvalidStateException.
    c = CircularChannel{Int}(1)
    t = Threads.@spawn try
        take!(c)
    catch ex
        ex
    end
    @test timedwait(() -> istaskstarted(t), 10) == :ok
    close(c)
    @test fetch(t) isa InvalidStateException

    # Multiple consumers: each put! is delivered to exactly one take!.
    # Capacity >= n ensures no drops, so every consumer receives an item.
    n = 50
    c = CircularChannel{Int}(n)
    consumers = [Threads.@spawn(take!(c)) for _ in 1:n]
    for i in 1:n
        put!(c, i)
    end
    taken = sort(fetch.(consumers))
    @test drop_count(c) == 0
    @test taken == collect(1:n)
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

            for _ in 1:XfaEngine.VARIABLE_CHANNEL_SIZE
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

@testset "Trainmatching" begin
    # Initialize the matcher to look for one source
    tm = Context.Trainmatcher(["foo.bar"], 2)
    data = VariableData(1, "foo.bar", 1)

    matched_trains = Context.match_train(tm, data)
    @test length(matched_trains) == 1
    @test only(keys(matched_trains[1])) == "foo.bar"

    # And multiple sources
    tm = Context.Trainmatcher(["foo.bar", "foo.baz"], 2)
    @test isempty(Context.match_train(tm, VariableData(1, "foo.bar", 1)))
    matched_trains = Context.match_train(tm, VariableData(1, "foo.baz", 1))
    @test length(matched_trains) == 1
    @test Set(keys(matched_trains[1])) == Set(["foo.bar", "foo.baz"])

    # Test the max train latency
    tm = Context.Trainmatcher(["foo.bar", "foo.baz"], 1)
    @test isempty(Context.match_train(tm, VariableData(1, "foo.bar", 1)))
    @test isempty(Context.match_train(tm, VariableData(3, "foo.bar", 1)))
    @test isempty(Context.match_train(tm, VariableData(1, "foo.baz", 1)))
    @test length(Context.match_train(tm, VariableData(3, "foo.baz", 1))) == 1
end

@testset "karabo_dependency" begin
    @test karabo"foo.bar" == karabo_dependency("foo", "bar")
    @test karabo"foo.bar.baz" == karabo_dependency("foo", "bar.baz")
    @test karabo"foo:output[bar]" == karabo_dependency("foo:output", "bar")
    @test karabo"foo:channel_1.output[bar]" == karabo_dependency("foo:channel_1.output", "bar")

    @test_throws ArgumentError karabo_dependency("foo")
    @test_throws ArgumentError karabo_dependency("foo.bar[]")
    @test_throws ArgumentError karabo_dependency("foo:[bar]")

    # Topic macros
    @test karabo"MID//foo.bar" == karabo_dependency("MID", "foo", "bar")
    @test karabo"SA2//foo:output[bar]" == karabo_dependency("SA2", "foo:output", "bar")

    # Parsing from string with topic
    @test karabo_dependency("MID//foo.bar") == karabo_dependency("MID", "foo", "bar")
    @test karabo_dependency("SA2//foo:output[bar]") == karabo_dependency("SA2", "foo:output", "bar")

    # Round trip
    @test karabo_dependency(string(karabo"MID//foo.bar")) == karabo"MID//foo.bar"
    @test karabo_dependency(string(karabo"SA2//foo:output[bar]")) == karabo"SA2//foo:output[bar]"

    # Proxy
    @test karabo"MID//foo.bar@px" == karabo_dependency("MID", "foo", "bar", "px")
    @test karabo_dependency(string(karabo"MID//foo:output[bar]@px")) == karabo"MID//foo:output[bar]@px"
end

# Helper module that defines variables for reference tests, defined in
# Main so that load_from_string's context modules can access it.
@eval Main module VariableLibrary
    using XfaEngine.Context
    @Variable function normalize(data -> karabo"camera.pixels")
        return data ./ maximum(data)
    end

    @Variable function with_subvar(data -> karabo"device.property")
        @add_subvariable("half", data / 2)
        return data
    end
end

# Test postprocessors, also in Main for load_from_string access.
@eval Main module PostprocessorLibrary
    using Statistics: mean
    using XfaEngine: Context
    using XfaEngine.Context: AbstractPostprocessor, Parameter

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

@testset "@Variable" begin
    # Smoke test for basic functionality
    ctx = Context.load_from_string("""
    using Statistics

    @Variable cam4 -> karabo"MID_EXP_SAM/CAM/CAM4:output[data.image.pixels]"

    @Variable function xgm(intensity -> karabo"SA2_XTD1_XGM/XGM/DOOCS:output[data.intensityTD]")
        return mean(intensity)
    end
    """)

    expected_variables = Set(["cam4", "xgm"])
    @test Set(keys(ctx.functions)) == expected_variables
    invokelatest() do
        @test ctx.functions["cam4"](10) == 10
        @test ctx.functions["xgm"](1:10) == mean(1:10)
    end

    @test Set(keys(ctx.dag)) == expected_variables

    @test ctx.dag["cam4"] == OD("data" => karabo"MID_EXP_SAM/CAM/CAM4:output[data.image.pixels]")
    @test ctx.dag["xgm"] == OD("intensity" => karabo"SA2_XTD1_XGM/XGM/DOOCS:output[data.intensityTD]")

    # Test generating variables dynamically
    ctx = Context.load_from_string(raw"""
    function xgm()
        for x in [:foo, :bar, :baz]
            @eval @Variable $x -> $(karabo"$x.data")
        end
    end

    xgm()
    """)

    # All the variables should have been generated
    expected_variables = Set(["foo", "bar", "baz"])
    @test Set(keys(ctx.functions)) == expected_variables

    # And their dependencies should have been marked
    for name in expected_variables
        @test ctx.dag[name] == OD("data" => karabo_dependency(name, "data"))
    end
    @test Context.external_dependencies(ctx; per_variable=true) == Dict("foo" => [karabo"foo.data"],
                                                                        "bar" => [karabo"bar.data"],
                                                                        "baz" => [karabo"baz.data"])

    # Test variables depending on each other
    ctx = Context.load_from_string("""
    @Variable foo -> karabo"foo.bar"

    @Variable function bar(data -> foo)
        data
    end
    """)

    @test ctx.dag["bar"] == OD("data" => Dependency("foo"))
    @test ctx.dag["foo"] == OD("data" => karabo"foo.bar")

    # Creating a short-hand variable pointing to anything other than a proper
    # dependency should fail. We test the internal function here because it's
    # easier to test than the macro evaluated at parse time.
    @test_throws ArgumentError Context._variable(@__MODULE__, :(foo -> 42), false)
    @test_throws ArgumentError Context._variable(@__MODULE__, :(foo -> "foo.bar"), false)

    # Using an unrecognized macro as a dependency should fail
    @test_throws ArgumentError Context._variable(@__MODULE__, :(function foo(data -> bar"baz") data end), false)

    # We should not be able to create a subvariable that isn't defined at the
    # top level of a function.
    @test_throws "defined at the toplevel" Context._variable(@__MODULE__, quote
                                                                 function foo()
                                                                     if true
                                                                         @add_subvariable("data", 42)
                                                                     end
                                                                 end
                                                             end,
                                                             false)

    # Test creating a subvariable
    ctx = Context.load_from_string("""
    @Variable function foo(data -> karabo"device.property")
        @add_subvariable("bar", mean(data))

        return data
    end

    @Variable function quux(data -> foo.bar)
        42
    end
    """)
    @test Set(keys(ctx.functions)) == Set(["foo", "quux"])
    @test ctx.subvariables["foo"] == ["foo.bar"]
    @test ctx.dag["quux"] == OD("data" => subvariable_dependency("foo", "bar"))

    # Test loading from a file
    ctx_code = """
    @Variable foo -> karabo"foo.bar"
    """
    ctx_from_str = Context.load_from_string(ctx_code)
    path, io = mktemp()
    write(io, ctx_code)
    close(io)
    @test Context.load_from_file(path).dag == ctx_from_str.dag

    @testset "@Variable references" begin
        # Test bare reference: @Variable VariableLibrary.normalize
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        @Variable VariableLibrary.normalize
        """)
        @test keyset(ctx.functions) == Set(["normalize"])
        @test ctx.dag["normalize"] == OD("data" => karabo"camera.pixels")

        # Test renamed reference: @Variable my_norm -> VariableLibrary.normalize
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        using .VariableLibrary: normalize
        @Variable my_norm -> normalize
        """)
        @test keyset(ctx.functions) == Set(["my_norm"])
        @test ctx.dag["my_norm"] == OD("data" => karabo"camera.pixels")

        # Test reference with dependency override
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        @Variable VariableLibrary.normalize(data -> karabo"other_camera.data")
        """)
        @test ctx.dag["normalize"] == OD("data" => karabo"other_camera.data")

        # Test renamed reference with dependency override
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        @Variable my_norm -> VariableLibrary.normalize(data -> karabo"other_camera.data")
        """)
        @test keyset(ctx.functions) == Set(["my_norm"])
        @test ctx.dag["my_norm"] == OD("data" => karabo"other_camera.data")

        # Test that the wrapper function delegates to the original
        invokelatest() do
            @test ctx.functions["my_norm"]([2, 4, 6]) == Main.VariableLibrary.normalize([2, 4, 6])
        end

        # Test that variable_origin points to the original
        invokelatest() do
            my_norm_func = ctx.functions["my_norm"]
            @test Context.variable_origin(my_norm_func) === Main.VariableLibrary.normalize
        end

        # Test subvariable remapping on rename
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        @Variable renamed -> VariableLibrary.with_subvar
        """)
        @test ctx.subvariables["renamed"] == ["renamed.half"]

        # Test that the original variable is excluded from the context when referenced
        ctx = Context.load_from_string("""
        using Main: VariableLibrary
        @Variable my_norm -> VariableLibrary.normalize
        @Variable foo -> karabo"foo.bar"
        """)
        @test Set(keys(ctx.functions)) == Set(["my_norm", "foo"])
    end
end

@testset "@postprocess" begin
    # Execution with mixed @add_subvariable and @postprocess
    ctx = Context.load_from_string("""
    using Main.PostprocessorLibrary: TestMean, TestWindow

    @Input function input(::Context.MockInput, output)
        put!(output, (0, Dict("foo" => Dict("bar" => [1, 2, 3]))))
    end
    x = Context.MockInput()

    @Variable function foo(data -> karabo"foo.bar")
        @postprocess(TestWindow(; size=2))
        @postprocess("avg", TestMean())
        return data
    end
    """)
    @test Set(ctx.subvariables["foo"]) == Set(["foo.avg", "foo.window"])
    @test ctx.parameters["foo.window.size"][] == 2
    @test issetequal(["foo.avg", "foo.window"], keys(ctx.postprocessors))
    @test ctx.variable_postprocessors["foo"] == ["foo.window", "foo.avg"]

    Context.run(ctx) do
        @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
    end
    result = take!(ctx.stream_output)
    @test result.subvariables["foo.window"] == VariableData(0, "foo.window", [1, 2])
    @test result.subvariables["foo.avg"] == VariableData(0, "foo.avg", 2.0)

    # Changing a postprocessor parameter should update its value, invoke the
    # update handler with the postprocessor object, and affect subsequent runs.
    ctx = Context.load_from_string("""
    using Main.PostprocessorLibrary: TestWindow

    next_input = Base.Event()
    param_value = -1

    @Input function input(::Context.MockInput, output)
        put!(output, (0, Dict("foo" => Dict("bar" => 1:10))))
        wait(next_input)
        put!(output, (1, Dict("foo" => Dict("bar" => 1:10))))
    end
    i = Context.MockInput()

    @Variable function foo(data -> karabo"foo.bar")
        @postprocess(TestWindow(Parameter(; name="", value=3, update_handler=(_, value) -> global param_value = value)))
        return data
    end
    """)
    @test ctx.parameters["foo.window.size"][] == 3

    Context.run(ctx) do
        r1 = take!(ctx.stream_output)
        @test r1.subvariables["foo.window"].data == 1:3

        Context.change_parameter(ctx, Parameter("foo.window.size", 5))
        mod = Context.worker_state.current_ctx_module
        @test ctx.parameters["foo.window.size"][] == 5
        @test mod.param_value == 5

        notify(mod.next_input)
        r2 = take!(ctx.stream_output)
        @test r2.subvariables["foo.window"].data == 1:5
    end
end

@testset "@display" begin
    # Global parameter reference
    ctx = Context.load_from_string(raw"""
    roi = Parameter(Context.RectROI())

    @Variable function img(data -> karabo"camera.data")
        @display roi
        return data
    end
    """)
    @test invokelatest(Context.variable_displays, ctx.functions["img"]) == ["roi"]

    # Group parameter reference; the stored name is qualified by the group
    # type, not the (not-yet-known) instantiated group name.
    ctx = Context.load_from_string(raw"""
    @Group struct Cam
        roi::Parameter{Context.RectROI} = Parameter(Context.RectROI())
    end

    @Variable function img(c::Cam)
        @display c.roi
        return 0
    end

    cam = Cam()
    """)
    img_func = only(f for (n, f) in ctx.functions if endswith(n, ".img"))
    @test invokelatest(Context.variable_displays, img_func) == ["Cam.roi"]

    # Multiple @display entries on one variable
    ctx = Context.load_from_string(raw"""
    a = Parameter(Context.RectROI())
    b = Parameter(Context.RectROI())

    @Variable function img(data -> karabo"camera.data")
        @display a
        @display b
        return data
    end
    """)
    @test invokelatest(Context.variable_displays, ctx.functions["img"]) == ["a", "b"]

    # `head.tail` where `head` isn't the group arg is rejected
    @test_throws "not the group argument" Context._variable(@__MODULE__, quote
        function img(c::Cam)
            @display other.roi
            return 0
        end
    end, false)

    # Non-symbol, non-qualified display expressions are rejected
    @test_throws "@display takes" Context._variable(@__MODULE__, quote
        function img(data -> karabo"camera.data")
            @display 1 + 1
            return data
        end
    end, false)

    # @display outside a @Variable block errors
    @test_throws "may only be used inside" eval(:(Context.@display foo))

    # Full context-load resolution: global ref stays as-is, group ref becomes
    # "$(instance).$(field)".
    ctx = Context.load_from_string(raw"""
    roi = Parameter(Context.RectROI())

    @Group struct Cam
        roi::Parameter{Context.RectROI} = Parameter(Context.RectROI())
    end

    @Variable function global_img(data -> karabo"camera.data")
        @display roi
        return data
    end

    @Variable function img(c::Cam)
        @display c.roi
        return 0
    end

    cam = Cam()
    """)
    @test ctx.displays["global_img"] == ["roi"]
    @test ctx.displays["cam.img"] == ["cam.roi"]

    # Reference to a missing parameter is rejected at load time
    @test_throws "unknown parameter" Context.load_from_string(raw"""
    @Variable function img(data -> karabo"camera.data")
        @display nope
        return data
    end
    """)

    # @display references propagate across @Variable references
    ctx = Context.load_from_string(raw"""
    using Main: VariableLibrary
    @Variable renamed -> VariableLibrary.normalize
    """)
    @test invokelatest(Context.variable_displays, ctx.functions["renamed"]) == String[]
end

@testset "Parameter" begin
    # Smoke tests for constructors
    @test Parameter(0) isa Parameter
    @test_throws ArgumentError Parameter(() -> 1, 0)
    @test Parameter(Returns(nothing), 0) isa Parameter
    @test Parameter("foo", 1) isa Parameter

    # Test creating top-level parameters
    ctx = Context.load_from_string(raw"""
    photon_energy = Parameter(0)
    device = Parameter("foo")
    """)
    @test ctx.parameters == Dict("photon_energy" => Parameter("photon_energy", 0),
                                 "device" => Parameter("device", "foo"))

    # Test assigning parameters
    ctx = Context.load_from_string(raw"""
    photon_energy = Parameter(0.0)

    @Input function input(_::Context.MockInput, output)
        put!(output, (0, Dict("camera" => Dict("data" => 42))))
    end

    x = Context.MockInput()

    @Variable function foo(data -> karabo"camera.data")
        tryset(photon_energy, 9)
        return data
    end
    """)
    log = TestLogger()
    with_logger(log) do
        Context.run(ctx) do
            @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
        end
    end
    # Waiting for the log message to come in is necessary because `tryset()`
    # uses `remote_do()` internally, which does not wait for the remotecall.
    @test timedwait(() -> length(log.logs) == 1, 5) == :ok
    @test occursin("Setting parameter", log.logs[1].message)

    # Variables and parameters with the same name doesn't work
    @test_throws ErrorException Context.load_from_string(raw"""
    foo = Parameter(0)

    @Variable foo -> karabo"foo.bar"
    """)
end

@testset "@Input" begin
    @test_throws ArgumentError Context._input(@__MODULE__, "foo", false)
    @test_throws ArgumentError Context._input(@__MODULE__, :(1 + 1), false)

    # Test a standalone input function
    ctx = Context.load_from_string(raw"""
    @Input function bridge(_::Context.MockInput, output)
        put!(output, 42)
    end

    x = Context.MockInput()
    """)
    @test isempty(ctx.dag)
    @test ctx.inputs["x.bridge"] == Dict("_" => group_dependency("x", Context.MockInput))

    # And a input function that's part of a group
    ctx = Context.load_from_string(raw"""
    @Group struct Foo end
    @Input function bridge(::Foo, output)
        put!(output, 42)
    end

    foo = Foo()
    """)
    @test haskey(ctx.inputs, "foo.bridge")

    # But not one with arbitrary arguments
    @test_throws XfaContextException Context._input(@__MODULE__,
                                                    quote
                                                        function foo(output, bar)
                                                            42
                                                        end
                                                    end, false)
end

@testset "@Group" begin
    @test_throws ArgumentError Context._group(@__MODULE__, "foo", false)
    @test_throws ArgumentError Context._group(@__MODULE__, :(1 + 1), false)
    @test_throws ArgumentError Context._group(@__MODULE__, :(@kwdef struct Foo end), false)

    ctx = Context.load_from_string(raw"""
    @Group struct Foo end

    @Variable function foo(::Foo)
        42
    end
    """)

    # Creating a group variable should add it to the group definitions
    @test length(ctx.group_types) > 1
    group_key = only(filter(x -> nameof(x) == :Foo, keys(ctx.group_types)))
    @test nameof.(ctx.group_types[group_key].variables) == [:foo]
    # But because a group object hasn't been created it shouldn't actually
    # schedule anything.
    @test isempty(ctx.dag)

    # Test instantiating a group
    ctx = Context.load_from_string(raw"""
    @Group struct Foo
        bar::Parameter{Int} = Parameter(42)
    end

    @Variable function foo(data::Foo)
        data.bar
    end

    foo_group = Foo()
    """)
    group_type = only(filter(x -> nameof(x) == :Foo, keys(ctx.group_types)))
    @test ctx.dag == Dict("foo_group.foo" => OD("data" => group_dependency("foo_group", group_type)))
    @test ctx.parameters == Dict("foo_group.bar" => Parameter("foo_group.bar", 42))

    # Test that @kwdef groups accept raw values for Parameter fields,
    # and that handlers from the default are preserved.
    ctx = Context.load_from_string(raw"""
    handler_called = Ref(false)
    @Group mutable struct Bar
        x::Parameter{Int} = Parameter(0) do _; handler_called[] = true end
        y::Parameter{Int}
        z::Int = 5
    end

    bar = Bar(; y=10)
    """)
    bar = ctx.groups["bar"]
    @test bar.x[] == 0 && bar.y[] == 10 && bar.z == 5
    @test !isnothing(bar.x.update_handler)
    @invokelatest bar.x.update_handler(99)
    @test invokelatest() do
        Context.worker_state.current_ctx_module.handler_called[]
    end

    # Test that the struct can be used as a dependency
    ctx = Context.load_from_string(raw"""
    @Group struct Foo
        value::Float64
    end

    @Variable function foo(data::Foo)
        data.value
    end

    foo_group = Foo(; value=2π)

    @Variable function bar(data -> foo_group.foo)
        data
    end
    """)
    @test ctx.dag["bar"] == OD("data" => Context.Dependency("foo_group.foo"))

    # Test that group variable dependencies must reference group parameters
    @test_throws ArgumentError Context._variable(@__MODULE__, :(function bar(::Foo, data -> karabo"motor1.pos") data end), false)
    @test_throws ArgumentError Context._variable(@__MODULE__, :(function bar(::Foo, data -> some_var) data end), false)

    # Test group parameter dependency resolution
    ctx = Context.load_from_string(raw"""
    @Group mutable struct Foo
        source::Parameter{Dependency}
    end

    @Variable function foo(group::Foo, data -> Foo.source)
        data
    end

    foo_group = Foo(; source=karabo"motor1.pos")
    """)
    @test ctx.dag["foo_group.foo"] == OD("group" => group_dependency("foo_group", only(filter(x -> nameof(x) == :Foo, keys(ctx.group_types)))),
                                         "data" => karabo"motor1.pos")

    # Test that referencing a non-existent parameter throws
    @test_throws XfaContextException Context.load_from_string(raw"""
    @Group mutable struct Foo end

    @Variable function foo(group::Foo, data -> Foo.nonexistent)
        data
    end

    foo_group = Foo()
    """)

    # Test instantiating groups from other modules
    helper_file_path = joinpath(@__DIR__, "dummy_variables.jl")
    ctx = Context.load_from_string("""
    Base.include(@__MODULE__, "$(helper_file_path)")

    bridge = KaraboBridge(; trainmatcher=KaraboDevice("MATCHER"))

    foo = DummyVariables.Foo(; bar=1)
    """)
    @test haskey(ctx.inputs, "bridge.stream")
    @test ctx.functions["bridge.stream"] === Context.stream
    @test haskey(ctx.dag, "foo.compute")
end

@testset "Scheduler" begin
    @testset "Topological sort" begin
        # Test sorting a DAG with a cycle
        dag = Dict("foo" => ["bar"], "bar" => ["foo"])
        @test_throws XfaContextException Context.topological_sort(dag)

        # Sort an empty DAG
        @test Context.topological_sort(Dict("foo" => [])) == ["foo"]

        # Test that external dependencies aren't considered during sorting
        dag = Dict("camera" => [karabo"foo.bar", karabo"baz.quux"])
        @test Context.topological_sort(dag) == ["camera"]

        # Subvariables should be ignored too
        dag = Dict("camera" => [], "foo" => [subvariable_dependency("camera", "bar")])
        @test Context.topological_sort(dag) == ["camera", "foo"]

        # Test that sorting actually works
        dag = Dict("camera" => [karabo"foo.bar"], "foo" => ["camera"], "bar" => ["foo"])
        @test Context.topological_sort(dag) == ["camera", "foo", "bar"]
    end

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
            bridge_a = KaraboBridge(; trainmatcher=KaraboDevice("T1//DEV_A"))
            bridge_a._mock_sources = String[]

            bridge_b = KaraboBridge(; trainmatcher=KaraboDevice("T2//DEV_B"))
            bridge_b._mock_sources = String[]

            @Variable foo -> karabo"foo.bar"
            @Variable special -> karabo"T1//special.src"
            """

            # No rules: topic-match routes the prefixed dep; unprefixed dep has no
            # topic/source match and two inputs exist, so it errors.
            @test_throws XfaContextException Context.load_from_string(ctx_src)

            # Rule forces source "foo" to bridge_b (device name DEV_B) regardless
            # of topic. The topicked dep falls through to the topic-match heuristic.
            rules = [RoutingRule("*", "foo", "DEV_B")]
            ctx = Context.load_from_string(ctx_src; routing_rules=rules)
            @test ctx.dep_to_input["foo.bar"] == "bridge_b.stream"
            @test ctx.dep_to_input["T1//special.src"] == "bridge_a.stream"

            # Rule pointing at a device that isn't among the inputs falls through
            # to the existing heuristics (the trailing rule keeps foo routable).
            rules = [RoutingRule("*", "special", "NONEXISTENT_DEV"),
                     RoutingRule("*", "*", "DEV_A")]
            ctx = Context.load_from_string(ctx_src; routing_rules=rules)
            @test ctx.dep_to_input["T1//special.src"] == "bridge_a.stream"

            # First-match-wins: a specific rule overrides the catch-all below it.
            rules = [RoutingRule("*", "foo", "DEV_A"),
                     RoutingRule("*", "*", "DEV_B")]
            ctx = Context.load_from_string(ctx_src; routing_rules=rules)
            @test ctx.dep_to_input["foo.bar"] == "bridge_a.stream"
            @test ctx.dep_to_input["T1//special.src"] == "bridge_b.stream"

            # Topic-qualified input ("T//DEV") disambiguates when multiple
            # topics have devices with the same name.
            same_name_src = raw"""
            bridge_a = KaraboBridge(; trainmatcher=KaraboDevice("T1//DEV"))
            bridge_a._mock_sources = String[]

            bridge_b = KaraboBridge(; trainmatcher=KaraboDevice("T2//DEV"))
            bridge_b._mock_sources = String[]

            @Variable foo -> karabo"foo.bar"
            """
            rules = [RoutingRule("*", "foo", "T2//DEV")]
            ctx = Context.load_from_string(same_name_src; routing_rules=rules)
            @test ctx.dep_to_input["foo.bar"] == "bridge_b.stream"

            rules = [RoutingRule("*", "foo", "T1//DEV")]
            ctx = Context.load_from_string(same_name_src; routing_rules=rules)
            @test ctx.dep_to_input["foo.bar"] == "bridge_a.stream"
        end
    end

    @testset "Execution" begin
        ctx = Context.load_from_string(raw"""
        @Variable camera -> karabo"camera.data"
        """)
        # Variables shouldn't be executed unless they have all their dependencies
        @test length(Context.execute_variables(ctx, Dict())) == 0
        @test Context.execute_variables(ctx, Dict("camera.data" => 1)) == Dict("camera" => 1)

        # Test that dependencies are passed correctly
        ctx = Context.load_from_string(raw"""
        norm = Parameter(1)
        @Variable foo -> karabo"foo.bar"
        @Variable function bar(data -> foo)
            return (2 * data, norm[])
        end
        """)
        @test Context.execute_variables(ctx, Dict("foo.bar" => 1)) == Dict("foo" => 1, "bar" => (2, 1))

        # Test executing inputs
        ctx = Context.load_from_string("""
        @Input function fakecamera(::Context.MockInput, output)
            tid = 0
            data = Dict("camera" => Dict("data" => rand(100, 100)))
            while true
                put!(output, (tid, data))
                tid += 1
            end
        end

        x = Context.MockInput()
        """)
        Context.run(ctx) do
            @test length(ctx.input_channels) == 1
            @test timedwait(() -> isready(ctx.input_channels["x.fakecamera"]), 10) == :ok

            @test isempty(ctx.input_variable_channels["x.fakecamera"])
        end
        @test istaskdone(ctx.input_tasks["x.fakecamera"])
        @test istaskdone(ctx.input_variables_tasks["x.fakecamera"])

        # Stopping execution should close all tasks/channels
        @test !isopen(ctx.stream_output)

        # Test executing external dependency variables
        ctx = Context.load_from_string("""
        @Input function fakecamera(::Context.MockInput, output)
            put!(output, (0, Dict("camera" => Dict("data" => 42))))
        end

        @Variable foo -> karabo"camera.data"

        x = Context.MockInput()
        """)
        Context.run(ctx) do
            @test only(keys(ctx.external_dependency_tasks)) == "camera.data"
            @test only(keys(ctx.external_dependency_channels["camera.data"])) == "foo"
            @test only(keys(ctx.variable_tasks)) == "foo"

            @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
        end
        @test istaskdone(ctx.external_dependency_tasks["camera.data"])
        @test take!(ctx.stream_output) == VariableData(0, "foo", 42)

        # Test executing variables
        ctx = Context.load_from_string("""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 1), "motor2" => Dict("pos" => 2))))
        end
        x = Context.MockInput()

        @Variable motor1 -> karabo"motor1.pos"

        @Variable function bar(motor1 -> motor1, motor2 -> karabo"motor2.pos")
            return motor1 + motor2
        end
        """)
        Context.run(ctx) do
            @test keys(ctx.variable_tasks) == Set(["motor1", "bar"])
            @test timedwait(() -> istaskdone(ctx.variable_tasks["bar"]), 5) == :ok
        end
        @test take!(ctx.stream_output) == VariableData(0, "motor1", 1)
        @test take!(ctx.stream_output) == VariableData(0, "bar", 3)

        # Variables that throw shouldn't cause execution of the other variables to
        # fail.
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 1))))
        end
        x = Context.MockInput()

        @Variable function foo(data -> karabo"motor1.pos")
            error("foo")
        end

        @Variable function bar(data -> karabo"motor1.pos")
            return data
        end
        """)
        log = TestLogger()
        with_logger(log) do
            Context.run(ctx) do
                @test timedwait(() -> isready(ctx.stream_output), 5) == :ok
            end
        end
        @test length(log.logs) == 1
        @test occursin("Execution of variable 'foo' failed", log.logs[1].message)
        @test take!(ctx.stream_output) == VariableData(0, "bar", 1)

        # Variables that fail should block downstream dependencies from running
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 1))))
        end
        x = Context.MockInput()

        @Variable function foo(data -> karabo"motor1.pos")
            error("foo")
        end

        @Variable function bar(data -> foo)
            return data
        end
        """)
        log = TestLogger()
        with_logger(log) do
            Context.run(ctx) do
                @test timedwait(() -> istaskdone(ctx.variable_tasks["bar"]), 5) == :ok
            end
        end
        @test length(log.logs) == 1
        @test !isready(ctx.stream_output)

        # Slightly more complicated DAG to test that everything is wired up correctly
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 1), "motor2" => Dict("pos" => 1))))
        end
        i = Context.MockInput()

        @Variable function x(data -> karabo"motor1.pos")
            return data
        end

        @Variable function y(data -> karabo"motor2.pos")
            return data
        end

        @Variable function z(x -> x, y -> y)
            return x + y
        end
        """)
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end

        # Take all the outputs
        results = VariableData[]
        while isready(ctx.stream_output)
            push!(results, take!(ctx.stream_output))
        end

        # Check that we have results from each variable
        @test length(results) == 3
        @test Set(results) == Set([VariableData(0, "x", 1),
                                   VariableData(0, "y", 1),
                                   VariableData(0, "z", 2)])

        # Test scheduling with groups and parameters
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 1))))
        end
        x = Context.MockInput()

        @Group struct Foo
            x::Parameter{Int}
            source::Parameter{Dependency}
        end

        @Variable function bar(group::Foo, data -> Foo.source)
            return group.x[] + data
        end

        foo = Foo(; x=1, source=karabo"motor1.pos")
        """)
        @test "foo.x" ∈ keys(ctx.parameters)
        @test "foo.source" ∈ keys(ctx.parameters)
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end
        @test isready(ctx.stream_output)
        @test take!(ctx.stream_output) == VariableData(0, "foo.bar", 2)

        # Test subvariable execution
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor1" => Dict("pos" => 10))))
        end
        x = Context.MockInput()

        @Variable function foo(data -> karabo"motor1.pos")
            @add_subvariable("half", data / 2)
            return data
        end

        @Variable function bar(data -> foo.half)
            return data + 1
        end
        """)
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end

        results = VariableData[]
        while isready(ctx.stream_output)
            push!(results, take!(ctx.stream_output))
        end
        @test length(results) == 2
        @test results[1] == VariableData(0, "foo", 10, Dict{String, Any}("foo.half" => VariableData(0, "foo.half", 5.0)))
        @test results[2] == VariableData(0, "bar", 6.0)

        # Test that returning a VariableData from a variable function overwrites
        # tid, name, and subvariables but preserves metadata fields.
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (5, Dict("motor1" => Dict("pos" => 10))))
        end
        x = Context.MockInput()

        @Variable function foo(data -> karabo"motor1.pos")
            @add_subvariable("half", data / 2)
            return VariableData(; data=data * 2, xlabel="my x", ylabel="my y",
                                x_axis=[1.0, 2.0, 3.0], y_axis=[1, 2, 3],
                                title="Foo", unit="j")
        end
        """)
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end
        result = take!(ctx.stream_output)
        @test result.tid == 5
        @test result.name == "foo"
        @test result.data == 20
        @test result.subvariables == Dict{String, Any}("foo.half" => VariableData(5, "foo.half", 5.0))
        @test result.xlabel == "my x"
        @test result.ylabel == "my y"
        @test result.x_axis == [1.0, 2.0, 3.0]
        @test result.y_axis == [1, 2, 3]
        @test result.title == "Foo"
        @test result.unit == "j"

        # Test input groups
        ctx = Context.load_from_string(raw"""
        @Group struct Foo
            x::Int
        end
        Context.update_sources(::Foo, _) = nothing

        @Input function input(foo::Foo, output)
            put!(output, (0, Dict("foo" => Dict("x" => foo.x))))
        end

        foo = Foo(; x=42)

        @Variable bar -> karabo"foo.x"
        """)
        @test only(keys(ctx.inputs)) == "foo.input"
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 2) == :ok
        end
        @test take!(ctx.stream_output) == VariableData(0, "bar", 42)

        # Test the Meta module
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (42, Dict("motor1" => Dict("pos" => 1))))
        end
        x = Context.MockInput()

        @Variable function foo(data -> karabo"motor1.pos")
            scratch = Meta.scratch[]

            return (; tid=Meta.tid[], scratch_dict=scratch isa Dict)
        end
        """)
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end
        result = take!(ctx.stream_output)
        @test result == VariableData(42, "foo", (; tid=42, scratch_dict=true))

        # Test changing parameters
        ctx = Context.load_from_string(raw"""
        next_input = Base.Event()

        @Input function input(::Context.MockInput, output)
            put!(output, (42, Dict("motor1" => Dict("pos" => 1))))
            wait(next_input)
            put!(output, (42, Dict("motor1" => Dict("pos" => 1))))
        end
        i = Context.MockInput()

        x_side_effect = 0
        x = Parameter(0) do x
            global x_side_effect = x
        end

        @Variable function foo(data -> karabo"motor1.pos")
            return x[]
        end
        """)
        Context.run(ctx) do
            @test take!(ctx.stream_output).data == 0
            Context.change_parameter(ctx, Parameter("x", 1))
            notify(Context.worker_state.current_ctx_module.next_input)
            @test take!(ctx.stream_output).data == 1
            @test Context.worker_state.current_ctx_module.x_side_effect == 1
        end

        # Test that group parameter update handlers receive the group object
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (42, Dict("motor1" => Dict("pos" => 1))))
        end
        i = Context.MockInput()

        @Group mutable struct MyGroup
            handler_received_value::Int = 0
            x::Parameter{Int} = Parameter(10) do group, value
                group.handler_received_value = value * 2
            end
        end

        g = MyGroup()

        @Variable function foo(_ -> karabo"motor1.pos")
            return g.x[]
        end
        """)
        Context.run(ctx) do
            @test take!(ctx.stream_output) == VariableData(42, "foo", 10)
            Context.change_parameter(ctx, Parameter("g.x", 5))
            @test ctx.groups["g"].handler_received_value == 10
            @test ctx.groups["g"].x[] == 5
        end
    end

    @testset "Multiple inputs" begin
        # Two inputs with different topics, deps routed by topic
        ctx = Context.load_from_string(raw"""
        @Group struct TopicA end
        Context.update_sources(::TopicA, _) = nothing
        Context.input_topic(::TopicA) = "SA2"

        @Group struct TopicB end
        Context.update_sources(::TopicB, _) = nothing
        Context.input_topic(::TopicB) = "MID"

        @Input function sa2_input(::TopicA, output)
            put!(output, (0, Dict("SA2_DEVICE" => Dict("val" => 10))))
        end

        @Input function mid_input(::TopicB, output)
            put!(output, (0, Dict("MID_DEVICE" => Dict("val" => 20))))
        end

        a = TopicA()
        b = TopicB()

        @Variable sa2_data -> karabo"SA2//SA2_DEVICE.val"
        @Variable mid_data -> karabo"MID//MID_DEVICE.val"
        """)
        @test ctx.dep_to_input["SA2//SA2_DEVICE.val"] == "a.sa2_input"
        @test ctx.dep_to_input["MID//MID_DEVICE.val"] == "b.mid_input"

        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end
        results = Dict{String, Any}()
        while isready(ctx.stream_output)
            r = take!(ctx.stream_output)
            results[r.name] = r.data
        end
        @test results["sa2_data"] == 10
        @test results["mid_data"] == 20

        # Two inputs with topics, dep without a topic should error
        @test_throws XfaContextException Context.load_from_string(raw"""
        @Group struct TopicA2 end
        Context.update_sources(::TopicA2, _) = nothing
        Context.input_topic(::TopicA2) = "SA2"

        @Group struct TopicB2 end
        Context.update_sources(::TopicB2, _) = nothing
        Context.input_topic(::TopicB2) = "MID"

        @Input function sa2_input(::TopicA2, output) end
        @Input function mid_input(::TopicB2, output) end

        a = TopicA2()
        b = TopicB2()

        @Variable foo -> karabo"unknown_device.val"
        """)

        # Test that single-input contexts still work without topics
        ctx = Context.load_from_string(raw"""
        @Input function input(::Context.MockInput, output)
            put!(output, (0, Dict("motor" => Dict("pos" => 42))))
        end
        x = Context.MockInput()

        @Variable motor_pos -> karabo"motor.pos"
        """)
        @test only(values(ctx.dep_to_input)) == "x.input"
        Context.run(ctx) do
            @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
        end
        @test take!(ctx.stream_output) == VariableData(0, "motor_pos", 42)
    end
end

@testset "Pipeline drops" begin
    # With a slow downstream variable, a fast producer should not block: items
    # are dropped in the variable channel rather than stalling upstream. Produce
    # many more trains than the channel capacity (100) and check that the slow
    # consumer processed fewer than were produced while the pipeline still ran
    # to completion.
    ctx = Context.load_from_string("""
    n_trains::Int = 500
    processed::Int = 0

    @Input function input(::Context.MockInput, output)
        for tid in 1:n_trains
            put!(output, (tid, Dict("motor" => Dict("pos" => tid))))
        end
    end
    x = Context.MockInput()

    @Variable function slow(data -> karabo"motor.pos")
        sleep(0.005)
        global processed += 1
        return data
    end
    """)
    Context.run(ctx) do
        @test timedwait(() -> !isopen(ctx.stream_output), 10) == :ok
    end

    mod = Context.worker_state.current_ctx_module
    n_processed = mod.processed[]
    @test 0 < n_processed < mod.n_trains

    # We should have stored the last 100 elements
    outputs = [x.data for x in ctx.stream_output]
    @test outputs == 401:500
end

@testset "Context builtins" begin
    @testset "Mean" begin
        # Reducing over all dims
        m = Context.Mean()
        @test m([1.0, 2.0, 3.0, NaN]) == 2.0
        @test isempty(m.buffer)

        # Reducing over specific dims with dropdims, with a NaN mixed in
        m = Context.Mean(; dims=(2,))
        A = [1.0 2.0 3.0; 4.0 NaN 6.0]
        @test m(A) == [2.0, 5.0]
        @test !isempty(m.buffer)
        buf = m.buffer

        # Calling again with matching type/dims reuses the buffer
        @test m(A .+ 1) == [3.0, 6.0]
        @test m.buffer === buf

        # Changing dims forces reallocation
        m.dims[] = Context.OptionalDims([1])
        @test m(A) == [2.5, 2.0, 4.5]
        @test m.buffer !== buf
    end

    @testset "MovingAvg" begin
        m = Context.MovingAvg(; nsamples=3)
        # First call seeds the buffer with the input promoted to float
        @test m([1, 2, 3]) == [1.0, 2.0, 3.0]
        @test eltype(m.buffer) === Float64
        buf = m.buffer

        # Subsequent calls with matching size/eltype reuse the buffer and apply
        # the EWMA in-place: alpha = 2/(3+1) = 0.5
        @test m([3, 4, 5]) == [2.0, 3.0, 4.0]
        @test m.buffer === buf

        # Changing size triggers reallocation
        @test m([1.0, 2.0]) == [1.0, 2.0]
        @test m.buffer !== buf

        # Changing eltype also triggers reallocation
        prev = m.buffer
        @test m(Float32[1, 2]) == Float32[1, 2]
        @test eltype(m.buffer) === Float32
        @test m.buffer !== prev
    end

    @testset "center_of_mass" begin
        # 1D: symmetric around index 2
        @test Context.center_of_mass([0.0, 1.0, 0.0]) == 2.0
        @test Context.center_of_mass([1.0, 0.0, 1.0]) == 2.0
        # Non-finite entries are skipped
        @test Context.center_of_mass([NaN, 1.0, 0.0, Inf]) == 2.0

        # 2D: returns (x=col, y=row); single hot pixel at (row=2, col=3)
        m = zeros(3, 4)
        m[2, 3] = 1.0
        @test Context.center_of_mass(m) == (3.0, 2.0)
    end

    @testset "Correlation" begin
        corr = Context.Correlation(; x=karabo"foo.bar", y=karabo"foo.baz")

        # compute_edges: empty samples → [-1,1]; otherwise min/max range
        @test Context.compute_edges((), 10) == -1:0.2:1

        cb1 = CircularBuffer{Float64}([1.0, 2.0])
        cb2 = CircularBuffer{Float64}([50.0, 100.0])
        @test Context.compute_edges(cb1, 10) == 1:0.1:2
        @test Context.compute_edges(Context.pulse_samples([cb1, cb2], [2]), 4) == 50:12.5:100
        @test Context.compute_edges(Context.pulse_samples([cb1, cb2], []), 4) == 1:24.75:100

        # Parameter update handlers should trigger rebuilding
        for handler in (corr.buffer_size.update_handler, corr.nbins.update_handler, corr.pulses.update_handler)
            corr.rebuild_histogram = false
            handler(corr, nothing)
            @test corr.rebuild_histogram
        end

        # update_buffer_size resizes existing buffers and invalidates
        push!(corr.x_buffers, CircularBuffer{Float64}(10))
        push!(corr.y_buffers, CircularBuffer{Float64}(10))
        corr.rebuild_histogram = false
        Context.update_buffer_size(corr, 500)
        @test capacity(corr.x_buffers[1]) == 500
        @test capacity(corr.y_buffers[1]) == 500
        @test corr.rebuild_histogram

        # Scalar inputs allocate a single per-pulse buffer at buffer_size
        corr = Context.Correlation(; x=karabo"foo.bar", y=karabo"foo.baz")
        corr.nbins[] = 10
        corr.buffer_size[] = 50
        Context.correlate(corr, 1.0, 2.0)
        @test length(corr.x_buffers) == 1
        @test capacity(corr.x_buffers[1]) == 50
        @test binedges(corr.histogram)[1] == -1:0.2:1

        # Vector inputs create one buffer per pulse; shrinking pops the extras
        corr = Context.Correlation(; x=karabo"foo.bar", y=karabo"foo.baz")
        Context.correlate(corr, [1.0, 2.0], [10.0, 20.0])
        @test length(corr.x_buffers) == 2
        Context.correlate(corr, [3.0], [30.0])
        @test length(corr.x_buffers) == 1

        # Changing nbins rebuilds the histogram with the new bin count
        corr = Context.Correlation(; x=karabo"foo.bar", y=karabo"foo.baz")
        corr.nbins[] = 10
        Context.correlate(corr, 1.0, 2.0)
        @test length(binedges(corr.histogram)[1]) == 11
        corr.nbins[] = 20
        corr.rebuild_histogram = true
        Context.correlate(corr, 2.0, 3.0)
        @test length(binedges(corr.histogram)[1]) == 21

        # `pulses` restricts which pulses contribute to edges and counts
        corr = Context.Correlation(; x=karabo"foo.bar", y=karabo"foo.baz")
        corr.pulses[] = [1]
        Context.correlate(corr, [1.0, 999.0], [2.0, 999.0])
        @test last(binedges(corr.histogram)[1]) ≤ 1.0
        @test sum(bincounts(corr.histogram)) == 1
    end

    @testset "Histogram1D" begin
        h = Context.Histogram1D(; buffer_size=100, nbins=10)

        # Fresh histogram: default edges, no counts
        @test length(bincenters(h)) == 10
        @test sum(bincounts(h)) == 0

        # Scalar push triggers an initial rebuild against the buffer range
        push!(h, 1.0)
        @test length(h.buffer) == 1
        @test sum(bincounts(h)) == 1
        @test !h.rebuild

        # append! a vector — each sample lands in the buffer and a bin
        append!(h, [2.0, 3.0, 4.0, 5.0])
        @test length(h.buffer) == 5
        @test sum(bincounts(h)) == 5
        # Edges should now span [1, 5] (just rebuilt by the buffer-sparse trigger)
        @test first(binedges(h.histogram)) == 1.0
        @test last(binedges(h.histogram)) == 5.0

        # Past the sparse threshold the hot path just pushes into existing bins,
        # leaving the edges alone until the periodic interval elapses.
        h = Context.Histogram1D(; buffer_size=100, nbins=10)
        append!(h, collect(1.0:30.0))
        edges_before = collect(binedges(h.histogram))
        append!(h, [100.0])
        @test collect(binedges(h.histogram)) == edges_before  # no rebuild
        @test sum(bincounts(h)) == 31

        # Forcing a stale interval triggers a rebuild with the new range
        h.last_edge_update = 0.0
        append!(h, [200.0])
        @test last(binedges(h.histogram)) == 200.0
        @test sum(bincounts(h)) == 32

        # Once the buffer fills, edges freeze and further samples accumulate
        # cumulatively without touching the buffer or rebuilding.
        h = Context.Histogram1D(; buffer_size=5, nbins=4)
        append!(h, [1.0, 2.0, 3.0, 4.0, 5.0])
        @test isfull(h.buffer)
        @test sum(bincounts(h)) == 5
        edges_at_fill = collect(binedges(h.histogram))
        buffer_at_fill = collect(h.buffer)

        # New samples bypass the buffer and bin against the frozen edges.
        append!(h, [10.0, 20.0])
        @test collect(h.buffer) == buffer_at_fill
        @test collect(binedges(h.histogram)) == edges_at_fill
        @test sum(bincounts(h)) == 7

        # Stale-driven rebuild no longer fires once the buffer is full.
        h.last_edge_update = 0.0
        append!(h, [3.0])
        @test collect(binedges(h.histogram)) == edges_at_fill
        @test sum(bincounts(h)) == 8

        # VariableData(::Histogram1D) wraps a histogram for direct return from
        # a variable: bincounts as data, bincenters as x_axis, :histogram type.
        h = Context.Histogram1D(; buffer_size=100, nbins=4)
        append!(h, [1.0, 2.0, 3.0, 4.0])
        vd = Context.VariableData(h; name="hist", xlabel="energy")
        @test vd.data == collect(bincounts(h))
        @test vd.x_axis == collect(bincenters(h))
        @test vd.plot_type === :histogram
        @test vd.name == "hist"
        @test vd.xlabel == "energy"
    end

    @testset "KaraboBridge" begin
        port = getavailableport(42000)
        address = "tcp://localhost:$(port)"
        bridge_server = KaraboBridgeServer(address)
        KaraboBridge.startbridge(bridge_server)

        ctx = Context.load_from_string("""
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("MATCHER"), sources=["foo.x"])
        bridge._mock_sources = String[]
        bridge.manual_configuration[] = true
        bridge.address[] = "$(address)"

        @Variable foo -> karabo"foo.x"
        """)

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
    parent = VariableData(; tid=0, name="p", data=[1, 2],
                          subvariables=Dict{String, Any}(
                              "p.scalar" => VariableData(0, "p.scalar", 1.5),
                              "p.arr" => VariableData(0, "p.arr", [4, 5])))
    f = build_client_view!(state, parent, sub(), cache())
    @test f.data isa ArrayMetadata
    @test keyset(f.subvariables) == Set(["p.scalar", "p.arr"])
    @test f.subvariables["p.scalar"].data == 1.5
    @test f.subvariables["p.arr"].data isa ArrayMetadata

    # Subscribing to the qualified subvariable name delivers the real array.
    f = build_client_view!(state, parent, sub("p.arr" => -1), cache())
    @test f.data isa ArrayMetadata
    @test f.subvariables["p.arr"].data == [4, 5]

    # Subscribing to the parent does not implicitly subscribe its subvariables.
    f = build_client_view!(state, parent, sub("p" => -1), cache())
    @test f.data == [1, 2]
    @test f.subvariables["p.arr"].data isa ArrayMetadata

    # Re-prepending the parent name (the historical bug) would look up
    # "p.p.arr" and miss the subscription — make sure that doesn't happen.
    f = build_client_view!(state, parent, sub("p.p.arr" => -1), cache())
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

        bridge = KaraboBridge(; trainmatcher=KaraboDevice(""))
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
        """)

    @test Context.to_dict(ctx) == Dict("inputs" => Dict("bridge.stream" => ["bridge"]),
                                       "groups" => ["bridge"],
                                       "dag" =>          Dict("xgm" => OD("data" => karabo"xgm.intensity"),
                                                              "foo" => OD(),
                                                              "bar" => OD("data" => Dependency("xgm"))),
                                       "subvariables" => Dict("xgm" => [],
                                                              "foo" => [],
                                                              "bar" => ["bar.max_data", "bar.window"]),
                                       "postprocessors" => Dict("bar" => ["bar.window"]),
                                       "displays" => Dict("bar" => ["roi"]),
                                       "origins" => Dict("xgm" => "xgm",
                                                         "foo" => "foo",
                                                         "bar" => "bar",
                                                         "bridge" => "XfaEngine.Context.KaraboBridge",
                                                         "bridge.stream" => "XfaEngine.Context.stream"),
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
end

@testset "ZfpWorkspace" begin
    ws = ZfpWorkspace()

    @testset "should_compress" begin
        @test should_compress(zeros(600))
        @test should_compress(rand(UInt8, 500))
        @test !should_compress(zeros(100))
        @test !should_compress("string")
        @test !should_compress(zeros(Bool, 600))
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
end

end
