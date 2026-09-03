module XfaContextTests

__revise_mode__ = :eval

# Copy CondaPkg.toml to the test project so that it gets found by CondaPkg
# during the tests. If this was instead in the project directory it would also
# be used by CondaPkg outside of the tests, which we don't want.
cp(joinpath(@__DIR__, "CondaPkg.toml"), joinpath(dirname(Base.active_project()), "CondaPkg.toml"); force=true)

ENV["JULIA_CONDAPKG_ENV"] = "@xfacontext-tests"
ENV["JULIA_CONDAPKG_VERBOSITY"] = -1

# If you're running the tests locally you could uncomment the two environment
# variables below. This will be a bit faster since it stops CondaPkg from
# re-resolving the environment each time (but you do need to run it at least
# once locally to initialize the environment).
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(Base.DEPOT_PATH[1], "conda_environments", "xfacontext-tests", "bin", "python")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using Statistics: mean
using Test: with_logger, TestLogger
using ReTest: @testset, @test, @test_throws

using OrderedCollections: OrderedDict as OD
using DataStructures: CircularBuffer, capacity, isfull
using FHist: bincounts, bincenters, binedges

using PythonCall

using XfaContext
using XfaContext: VariableData, XfaContextException, CircularChannel, drop_count

keyset(dict) = Set(keys(dict))

# Helper module that defines variables for reference tests, defined in
# Main so that load_from_string's context modules can access it.
@eval Main module VariableLibrary
    using XfaContext
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

# A registered @Input group for the available_variables test.
module InputLibrary
    using XfaContext

    @Group struct TestSource
        rate::Parameter{Int} = 10
    end

    @Input function test_stream(::TestSource, output)
        return nothing
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

@testset "available_variables" begin
    by_name = Dict(s.name => s for s in Context.available_variables())

    # A registered @Variable, with its arg dependency and subvariables.
    normalize = by_name["normalize"]
    @test normalize.kind == Context.VariableKind_Variable
    @test normalize.origin == "Main.VariableLibrary.normalize"
    @test normalize.dependencies["data"].kind == DepKind_Karabo
    @test by_name["with_subvar"].subvariables == ["with_subvar.half"]

    # A builtin @Group, with its Parameter fields and their defaults captured
    # (non-Parameter fields like `histogram` are excluded).
    correlation = by_name["Correlation"]
    @test correlation.kind == Context.VariableKind_Group
    @test correlation.group_parameters[:x] isa Parameter{Dependency}
    @test correlation.group_parameters[:buffer_size].value == 10_000
    @test !haskey(correlation.group_parameters, :histogram)
    # Handlers are stripped for wire safety.
    @test isnothing(correlation.group_parameters[:buffer_size].update_handler)

    # A @Group backing an @Input is reported as an input, not a plain group.
    testsource = by_name["TestSource"]
    @test testsource.kind == Context.VariableKind_Input
    @test haskey(testsource.group_parameters, :rate)

    # Group members (variables and inputs) aren't listed on their own.
    @test !haskey(by_name, "correlate") && !haskey(by_name, "test_stream")
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

@testset "ROIs" begin
    stack = reshape(1:40, 4, 5, 2)
    @test !isassigned(Context.RectROI())
    # Coordinates are rounded and clamped, trailing dimensions kept in full
    @test Context.RectROI(1.6, -3, 1, 100)(stack) == stack[1:4, 2:3, :]

    @test !isassigned(Context.LinearROI())
    @test_throws ArgumentError Context.LinearROI(; axis=:z)
    @test Context.LinearROI(2, 2)(stack) == stack[:, 2:4, :]
    @test Context.LinearROI(2, 1; axis=:y)(stack) == stack[2:3, :, :]
    # The axis is irrelevant for vectors
    @test Context.LinearROI(2.4, 1; axis=:y)(1:5) == 2:3
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

@testset "Property monitoring" begin
    watched_group = raw"""
    @Group mutable struct Watched
        device::Parameter{KaraboDevice} = Parameter{KaraboDevice}(; optional=true)
        updates::Vector{Any} = []
    end

    function Context.monitored_properties(w::Watched)
        if !isassigned(w.device)
            return Dependency[]
        end
        return [karabo_dependency(w.device[].name, "foo.bar"),
                karabo_dependency(w.device[].name, "baz")]
    end

    Context.on_properties_changed(w::Watched, changed) = push!(w.updates, changed)
    """

    ctx = Context.load_from_string("""
    $(watched_group)

    @Input function input(::Context.MockInput, output)
        # (train, foo.bar update tid, baz value, baz update tid). The first train
        # carries the trainmatcher's placeholder for a not-yet-valued property.
        for (tid, bar_tid, baz, baz_tid) in ((1, 0, nothing, 0), (2, 0, "b0", 0), (3, 2, "b0", 0), (4, 3, "b3", 3))
            put!(output, (tid, Dict("DEV" => Dict("foo.bar.value" => bar_tid * 10,
                                                   "foo.bar.timestamp.tid" => bar_tid,
                                                   "baz.value" => baz,
                                                   "baz.timestamp.tid" => baz_tid))))
        end
    end
    x = Context.MockInput()
    w = Watched(; device=KaraboDevice("", "DEV"))
    """)

    # Monitored deps are routed like DAG deps but get no channels of their own
    @test Set(string.(Context.external_dependencies(ctx))) == Set(["DEV.foo.bar", "DEV.baz"])
    @test isempty(Context.external_dependencies(ctx; monitored=false))
    @test ctx.dep_to_input == Dict("DEV.foo.bar" => "x.input", "DEV.baz" => "x.input")

    # The callback fires once on first receipt (`nothing` placeholders are
    # skipped until a value arrives), then only for properties whose update
    # tid changed, batched per train.
    w = ctx.groups["w"]
    Context.run(ctx) do
        @test timedwait(() -> length(w.updates) == 4, 5) == :ok
    end
    @test w.updates == [Dict("foo.bar" => (; value=0, tid=0)),
                        Dict("baz" => (; value="b0", tid=0)),
                        Dict("foo.bar" => (; value=20, tid=2)),
                        Dict("foo.bar" => (; value=30, tid=3), "baz" => (; value="b3", tid=3))]

    # Nothing to monitor while the device is unset
    ctx = Context.load_from_string("""
    $(watched_group)
    w = Watched()
    """)
    @test isempty(Context.external_dependencies(ctx))
end

@testset "Rewiring" begin
    follower = raw"""
    @Group mutable struct Follower
        device::Parameter{KaraboDevice} = Parameter{KaraboDevice}(; optional=true)
        source::Parameter{Dependency} = Parameter{Dependency}(; optional=true)
    end

    function Context.monitored_properties(f::Follower)
        if !isassigned(f.device)
            return Dependency[]
        end
        return [karabo_dependency(f.device[].name, "target")]
    end

    # Only report a change when the derived dependency actually changed, the
    # input replays `target` after every restart.
    function Context.on_properties_changed(f::Follower, changed)
        new_source = karabo_dependency(changed["target"].value, "pos")
        dep_changed = f.source[] != new_source
        tryset(f.source, new_source; force=true)
        return dep_changed
    end

    @Variable function follow(::Follower, data -> Follower.source)
        return data
    end

    @Input function input(::Context.MockInput, output)
        put!(output, (1, Dict("CTRL" => Dict("target.value" => "motor", "target.timestamp.tid" => 1))))
        for tid in 2:4
            put!(output, (tid, Dict("motor" => Dict("pos" => tid))))
        end
    end
    x = Context.MockInput()
    """
    follow_outputs(ctx) = [v.data for v in ctx.stream_output if v.name == "f.follow"]

    # Manual rewire: an unset optional dependency is a Parameter placeholder in
    # the DAG until the group assigns it and the context is rewired.
    ctx = Context.load_from_string("""
    $(follower)
    f = Follower()
    """)
    @test ctx.dag["f.follow"]["data"] isa Parameter
    @test isempty(ctx.dep_to_input)
    Context.run(ctx) do
        @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
    end

    Context.tryset(ctx.groups["f"].source, karabo_dependency("motor", "pos"); force=true)
    Context.rewire!(ctx)
    @test ctx.dag["f.follow"]["data"] == karabo_dependency("motor", "pos")
    @test ctx.dep_to_input == Dict("motor.pos" => "x.input")
    Context.run(ctx) do
        @test timedwait(() -> !isopen(ctx.stream_output), 5) == :ok
    end
    @test follow_outputs(ctx) == [2, 3, 4]

    # Automatic rewire: the first train derives the dependency and asks for a
    # restart, its replay after the restart doesn't.
    ctx = Context.load_from_string("""
    $(follower)
    f = Follower(; device=KaraboDevice("", "CTRL"))
    """)
    @test ctx.dep_to_input == Dict("CTRL.target" => "x.input")
    restarts = 0
    while Context.run_pipeline(ctx)
        restarts += 1
    end
    @test restarts == 1
    @test ctx.dag["f.follow"]["data"] == karabo_dependency("motor", "pos")
    @test ctx.dep_to_input == Dict("CTRL.target" => "x.input", "motor.pos" => "x.input")
    @test follow_outputs(ctx) == [2, 3, 4]

    # A stop request wins over a rewire request
    Context.start_pipeline(ctx)
    Context.request_rewire(ctx)
    Context.request_stop(ctx)
    @test !Context.wait_pipeline(ctx)
end

@testset "Scan automatic mode" begin
    ctx = Context.load_from_string(raw"""
    # The scantool's properties and the trains following them, edited by the test
    scantool = Ref{Any}(nothing)
    trains = Ref{Any}([])
    @Input function input(::Context.MockInput, output)
        put!(output, (1, Dict("SCAN" => scantool[])))
        for (tid, data) in trains[]
            put!(output, (tid, data))
        end
    end
    x = Context.MockInput()
    scn = Context.Scan(; value=karabo"det.intensity", scantool=KaraboDevice("", "SCAN"),
                       motor_priority=["MOTOR_A"])
    """)
    scn = ctx.groups["scn"]
    mod = Context.worker_state.current_ctx_module
    configure(props...) = mod.scantool[] = Dict(Iterators.flatten(("$(k).value" => v, "$(k).timestamp.tid" => 1)
                                                                  for (k, v) in props))
    train(tid, motor, pos, intensity) = tid => Dict("det" => Dict("intensity" => intensity),
                                                     motor => Dict("actualPosition" => pos))
    scan_outputs() = [v for v in ctx.stream_output if v.name == "scn.scan"]

    # Nothing is mirrored while the scantool's state is incomplete
    @test !Context.on_properties_changed(scn, Dict("scanEnv.scanType" => (; value="ascan", tid=1)))
    @test !isassigned(scn.position1)

    # A 1-D scan moving MOTOR_X and MOTOR_A together from a shared start point,
    # the latter in steps of 0.1: its first train derives the preferred motor and
    # restarts the pipeline, which then bins at half that motor's step size.
    configure("deviceEnv.activeMotorDeviceIds" => ["MOTOR_X", "MOTOR_A"], "deviceEnv.activeMotors" => ["x", "a"],
              "scanEnv.scanType" => "ascan", "scanEnv.startPoints" => [0.0],
              "scanEnv.stopPoints" => [5.0, 1.0], "scanEnv.steps" => [10])
    mod.trains[] = [train(2, "MOTOR_A", 0.0, 1.0), train(3, "MOTOR_A", 0.1, 2.0),
                    train(4, "MOTOR_A", 0.1, 3.0), train(5, "MOTOR_A", 0.2, 4.0)]
    @test Context.run_pipeline(ctx)
    @test scn.position1[] == karabo_dependency("MOTOR_A", "actualPosition")
    @test scn.resolution[] ≈ [0.05]
    @test !Context.run_pipeline(ctx)
    out = last(scan_outputs())
    @test Context.DD.lookup(out.data, :position1) ≈ [0.0, 0.1, 0.2]
    @test collect(out.data) ≈ [1.0, 2.5, 4.0]
    @test out.xlabel == "a"
    position = out.subvariables["scn.scan.position1"]
    @test position.data == 0.2 && position.title == "a" && position.bin_resolution ≈ 0.05

    # A scan of MOTOR_B alone rewires to it and resets the bins
    configure("deviceEnv.activeMotorDeviceIds" => ["MOTOR_B"], "deviceEnv.activeMotors" => ["b"],
              "scanEnv.scanType" => "dscan", "scanEnv.startPoints" => [0.0],
              "scanEnv.stopPoints" => [2.0], "scanEnv.steps" => [4])
    mod.trains[] = [train(2, "MOTOR_B", 0.0, 5.0)]
    @test Context.run_pipeline(ctx)
    @test !Context.run_pipeline(ctx)
    @test ctx.dag["scn.scan"]["position1"] == karabo_dependency("MOTOR_B", "actualPosition")
    @test scn.resolution[] ≈ [0.25]
    out = only(scan_outputs())
    @test collect(out.data) == [5.0]
    @test out.xlabel == "b"
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

    # A group variable can reference other @Variable's (and their
    # subvariables) of its own group through the GroupType name, alongside
    # Parameter fields.
    ctx = Context.load_from_string(raw"""
    @Group struct Foo end

    @Variable function bar(::Foo)
        @add_subvariable("sub", 1)
        0
    end

    @Variable function baz(::Foo, whole -> Foo.bar, part -> Foo.bar.sub)
        (whole, part)
    end

    f = Foo()
    """)
    @test ctx.dag["f.baz"]["whole"] == Context.Dependency("f.bar")
    @test ctx.dag["f.baz"]["part"] == subvariable_dependency("f.bar", "sub")

    # Referencing a non-existent member through the GroupType throws
    @test_throws XfaContextException Context.load_from_string(raw"""
    @Group struct Foo end

    @Variable function bar(::Foo, data -> Foo.missing) data end

    f = Foo()
    """)

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

    # Subvariable dependency passed via a group Parameter{Dependency}. The
    # dotted name in `Dependency("bar.sub")` must be resolved as a
    # subvariable so topological_sort can match it against the parent node.
    ctx = Context.load_from_string(raw"""
    @Group mutable struct Foo
        source::Parameter{Dependency}
    end

    @Variable function bar()
        42
    end

    @Variable function foo(group::Foo, data -> Foo.source)
        data
    end

    foo_group = Foo(; source=Dependency("bar.sub"))
    """)
    @test ctx.dag["foo_group.foo"]["data"] == subvariable_dependency("bar", "sub")

    # An unset optional Parameter{Dependency} is an optional dependency: it's
    # kept in the DAG (so the variable's positional args line up) as the
    # Parameter itself, but dropped from the serialized DAG edges. The parameter
    # is still registered (with its `optional` flag) so the client can draw it.
    ctx = Context.load_from_string(raw"""
    @Group mutable struct Foo
        source::Parameter{Dependency} = Parameter{Dependency}(; optional=true)
    end

    @Variable function foo(group::Foo, data -> Foo.source)
        data
    end

    foo_group = Foo()
    """)
    @test !isassigned(ctx.parameters["foo_group.source"])
    @test ctx.dag["foo_group.foo"]["data"] === ctx.groups["foo_group"].source
    serialized = Context.to_dict(ctx)
    @test !haskey(serialized["dag"]["foo_group.foo"], "data")
    @test serialized["parameters"]["foo_group.source"].optional

    # Wiring an optional dependency makes it a regular edge, and a raw value
    # passed to the constructor keeps the flag.
    ctx = Context.load_from_string(raw"""
    @Group mutable struct Foo
        source::Parameter{Dependency} = Parameter{Dependency}(; optional=true)
    end

    @Variable function foo(group::Foo, data -> Foo.source)
        data
    end

    foo_group = Foo(; source=karabo"motor1.pos")
    """)
    @test ctx.dag["foo_group.foo"]["data"] == karabo"motor1.pos"
    @test ctx.parameters["foo_group.source"].optional

    # A required dependency left unset is an error, not an implicit optional.
    @test_throws XfaContextException Context.load_from_string(raw"""
    @Group mutable struct Foo
        source::Parameter{Dependency} = Parameter{Dependency}()
    end

    @Variable function foo(group::Foo, data -> Foo.source)
        data
    end

    foo_group = Foo()
    """)

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

    foo = DummyVariables.Foo(; bar=1)
    """)
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

        # Scalars are averaged the same way
        m = Context.MovingAvg(; nsamples=3)
        @test m(1) === 1.0
        @test m(3) === 2.0

        # Non-finite samples are skipped, and don't poison an unseeded entry
        m = Context.MovingAvg(; nsamples=3)
        @test isequal(m([1.0, NaN]), [1.0, NaN])
        @test m([NaN, 4.0]) == [1.0, 4.0]
        @test m([3.0, Inf]) == [2.0, 4.0]
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

    @testset "VectorHistory" begin
        s = Context.VectorHistory(2; max_len=3)
        @test Context.capacity(s) == 3
        @test length(s) == 0

        for i in 1:5
            push!(s, [i, 10i])
        end
        # Only the last max_len vectors are retained, in order, as a view.
        @test length(s) == 3
        @test Context.data(s) == [3 30; 4 40; 5 50]
        @test s[3, :] == [5, 50]

        @test_throws ArgumentError push!(s, [1, 2, 3])

        Context.reset!(s)
        @test length(s) == 0
        @test isempty(Context.data(s))

        # average_window folds multiple pushes into one committed row.
        a = Context.VectorHistory(1; max_len=4, average_window=2)
        append!(a, ([0.0], [10.0], [4.0], [8.0]))
        @test length(a) == 2
        @test Context.data(a) == [5.0; 6.0;;]
    end

    @testset "BinnedSequence" begin
        @test_throws ArgumentError Context.Scalar2dScan([0.1])  # wrong res count

        # 1-D scalar binning: nearby positions merge, far ones open new levels,
        # and a revisit folds into the existing bin rather than splitting it.
        s = Context.Scalar1dScan(0.005)
        bin0_x, bin0_y = [-0.580, -0.581, -0.580], [10.0, 20.0, 30.0]
        bin1_x, bin1_y = [-0.570], [100.0]
        # Interleaved so the third bin-0 sample is a revisit after bin 1 opened.
        append!(s, bin0_x[1], bin0_y[1])
        append!(s, bin0_x[2], bin0_y[2])
        append!(s, bin1_x[1], bin1_y[1])
        append!(s, bin0_x[3], bin0_y[3])
        @test length(s) == 2
        @test Context.positions(s, 1) ≈ [mean(bin0_x), mean(bin1_x)]
        @test s.mean ≈ [mean(bin0_y), mean(bin1_y)]
        @test s.count == [length(bin0_y), length(bin1_y)]
        # Per-bin spread is the population std over the bin's samples, 0 for a singleton.
        @test s.std ≈ [sqrt(mean(abs2, bin0_y .- mean(bin0_y))), 0.0]

        # Discovery order != sorted order: a level found last but lying between
        # existing ones is placed correctly in the readout.
        s = Context.Scalar1dScan(0.5)
        xs, ys = [0.0, 2.0, 1.0], [1.0, 3.0, 2.0]
        for (x, y) in zip(xs, ys)
            append!(s, x, y)
        end
        order = sortperm(xs)
        @test Context.positions(s, 1) ≈ xs[order]
        @test s.mean ≈ ys[order]

        # 2-D raster of a scalar: A swept while B steps. The repeated A-sweep
        # reuses A-levels, giving a clean grid.
        as, bs = [0.0, 0.1, 0.2], [10.0, 20.0]
        g = Context.Scalar2dScan(0.05, 5.0)
        expected = zeros(length(as), length(bs))
        for (j, b) in enumerate(bs), (i, a) in enumerate(as)
            v = 100.0 * (j - 1) + i
            append!(g, (a, b), v)
            expected[i, j] = v
        end
        @test length(g) == length(as) * length(bs)
        @test Context.positions(g, 1) ≈ as
        @test Context.positions(g, 2) ≈ bs
        @test g.mean == expected
        @test g.count == fill(1, length(as), length(bs))

        # N-D (image) value: per-element nan-aware running mean; a NaN element is
        # skipped rather than dragged into the average.
        img = Context.BinnedSequence{2, 1}([1.0])
        f1, f2 = [1.0 2.0; 3.0 4.0], [3.0 NaN; 5.0 6.0]
        append!(img, 0.0, f1)
        append!(img, 0.2, f2)   # same bin (within res)
        means = img.mean
        @test size(means) == (1, size(f1)...)
        @test means[1, :, :] == map((a, b) -> isnan(b) ? a : mean((a, b)), f1, f2)

        # max_bins caps bin creation.
        capped = Context.Scalar1dScan(0.1; max_bins=2)
        for x in (0.0, 1.0, 2.0)
            append!(capped, x, x)
        end
        @test length(capped) == 2

        # reset! + replay reproduces the same bins (the resolution-change path).
        rebuilt = Context.Scalar1dScan(0.05)
        samples = [(-0.580, 10.0), (-0.581, 20.0), (-0.570, 100.0), (-0.580, 30.0)]
        for (x, y) in samples
            append!(rebuilt, x, y)
        end
        before = copy(rebuilt.mean)
        Context.reset!(rebuilt)
        @test length(rebuilt) == 0
        for (x, y) in samples
            append!(rebuilt, x, y)
        end
        @test rebuilt.mean == before
    end

    @testset "Scan" begin
        # The scan body reads Meta.name/subvariables for @add_subvariable, so bind
        # them and hand back the captured subvariable dict alongside the output.
        runscan(scn, args...) = begin
            subvars = Dict{String, Any}()
            out = Base.ScopedValues.@with(Context.Meta.name => "scan",
                                          Context.Meta.subvariables => subvars,
                                          Context.scan(scn, args...))
            (out, subvars)
        end

        # 1-D scalar scan: nearby positions merge, a revisit folds into the
        # existing bin, and the result is a DimArray over the bin positions with a
        # `counts` subvariable over the same grid.
        scn = Context.Scan(; value=karabo"foo.value", position1=karabo"mono.energy")
        scn.resolution[] = [0.005]
        local out, subvars
        for (p, v) in [(-0.580, 10.0), (-0.581, 20.0), (-0.570, 100.0), (-0.580, 30.0)]
            out, subvars = runscan(scn, v, p, nothing)
        end
        @test out isa VariableData && out.data isa Context.DD.DimArray && !out.compress
        @test Context.DD.lookup(out.data, :position1) ≈ [mean([-0.580, -0.581, -0.580]), -0.570]
        @test collect(out.data) ≈ [mean([10.0, 20.0, 30.0]), 100.0]
        @test out.xlabel == "mono.energy"
        @test collect(subvars["scan.counts"].data) == [3, 1]
        position = subvars["scan.position1"]
        @test position.data == -0.580 && position.title == "mono.energy" && position.bin_resolution == 0.005

        # 2-D scalar raster: wiring position2 adds a second axis, giving a clean
        # grid with one position lookup per motor.
        scn = Context.Scan(; value=karabo"foo.value",
                           position1=karabo"a.pos", position2=karabo"b.pos")
        scn.resolution[] = [0.05, 5.0]
        xs, ys = [0.0, 0.1, 0.2], [10.0, 20.0]
        for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
            out, subvars = runscan(scn, 100.0 * (j - 1) + i, x, y)
        end
        @test size(out.data) == (length(xs), length(ys))
        @test Context.DD.lookup(out.data, :position1) ≈ xs
        @test Context.DD.lookup(out.data, :position2) ≈ ys
        @test out.ylabel == "b.pos"

        # A resolution change replays the recompute buffer: coarsening the
        # resolution re-bins the retained tail, merging the two levels into one.
        scn = Context.Scan(; value=karabo"foo.value", position1=karabo"a.pos")
        scn.resolution[] = [0.005]
        for (p, v) in [(0.0, 1.0), (0.01, 3.0)]
            runscan(scn, v, p, nothing)
        end
        @test length(scn.binner.seq) == 2
        scn.resolution[] = [1.0]
        Context.rebin(scn, scn.resolution[])
        out, _ = runscan(scn, 5.0, 0.02, nothing)
        @test length(scn.binner.seq) == 1
        @test collect(out.data) ≈ [mean([1.0, 3.0, 5.0])]
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

        # Windowed mode: buffer_size is in trains, so the sample buffer is
        # sized to length(xs) * buffer_size on each append. Older trains drop
        # out as new ones arrive.
        h = Context.Histogram1D(; buffer_size=2, nbins=4, windowed=true)
        append!(h, [1.0, 2.0, 3.0])
        @test h.buffer.capacity == 6
        @test collect(h.buffer) == [1.0, 2.0, 3.0]
        @test sum(bincounts(h)) == 3

        append!(h, [4.0, 5.0, 6.0])
        @test isfull(h.buffer)
        @test collect(h.buffer) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        append!(h, [7.0, 8.0, 9.0])
        @test collect(h.buffer) == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        @test sum(bincounts(h)) == 6
        @test first(binedges(h.histogram)) == 4.0
        @test last(binedges(h.histogram)) == 9.0

        # Bumping buffer_size grows capacity on the next append, preserving
        # the most recent samples.
        h.buffer_size[] = 3
        append!(h, [10.0, 11.0, 12.0])
        @test h.buffer.capacity == 9
        @test collect(h.buffer) == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        @test sum(bincounts(h)) == 9
        @test last(binedges(h.histogram)) == 12.0
    end
end

@testset "Detector assembly" begin
    @test_throws DimensionMismatch Context.assemble!(zeros(3, 3), Context.AssemblerLUT(UInt64[1, 2, 3, 4], (2, 2)), rand(4))

    # Assembling from a vector of per-module arrays matches the equivalent
    # whole-detector array (single frame and a stack of frames).
    asm = Context.AssemblerLUT(UInt64[4, 3, 2, 1], (2, 2))
    frame = rand(2, 2)
    @test isequal(Context.assemble(asm, [frame[:, 1:1], frame[:, 2:2]]), Context.assemble(asm, frame))
    stack = rand(2, 2, 3)
    @test isequal(Context.assemble(asm, [stack[:, 1, :], stack[:, 2, :]]), Context.assemble(asm, stack))

    # ReTest may run this body on a migrated task, so hold the GIL around all
    # the PythonCall work to avoid calling into CPython without it.
    PythonCall.GIL.@lock begin
        eg = pyimport("extra_geom")
        python = pyconvert(String, pyimport("sys").executable)

        # XfaEngine works with dim-reversed arrays (e.g. (nfs, nss, nmod) rather
        # than numpy's (nmod, nss, nfs)), so flip dimensions when crossing to or
        # from extra-geom's numpy arrays.
        revdims(a) = permutedims(a, ndims(a):-1:1)

        # For each detector, check assembly matches extra-geom's own
        # position_modules_fast pixel for pixel (gaps included). ePix100 and
        # single-module JUNGFRAU have fixed layouts the script builds without a
        # geom file; the others are written to a .geom file and loaded back.
        detectors = ["AGIPD_1MGeometry", "DSSC_1MGeometry", "LPD_1MGeometry",
                     "JUNGFRAUGeometry", "Epix100Geometry"]
        mktempdir() do dir
            for name in detectors
                cls = getproperty(eg, Symbol(name))
                loaded, asm = if name == "Epix100Geometry"
                    (cls.pair_geometry(), Context.AssemblerLUT(name; python))
                elseif name == "JUNGFRAUGeometry"
                    (cls.from_module_positions(), Context.AssemblerLUT(name; python))
                else
                    geomfile = joinpath(dir, "$(name).geom")
                    cls.example().write_crystfel_geom(geomfile)
                    (cls.from_crystfel_geom(geomfile), Context.AssemblerLUT(name; geom_file=geomfile, python))
                end
                nmod, nss, nfs = pyconvert(Tuple{Int, Int, Int}, loaded.expected_data_shape)

                # Test single frame
                frame = rand(nfs, nss, nmod)
                ref = loaded.position_modules_fast(Py(revdims(frame)).to_numpy())[0]
                expected = revdims(pyconvert(Array{Float64}, ref))
                got = Context.assemble(asm, frame)
                @test size(got) == asm.frame_size
                @test isequal(got, expected)

                # Test multiple frames
                stack = rand(nfs, nss, nmod, 3)
                ref = loaded.position_modules_fast(Py(revdims(stack)).to_numpy())[0]
                expected_stack = revdims(pyconvert(Array{Float64}, ref))
                @test isequal(Context.assemble(asm, stack), expected_stack)
            end
        end
    end
end

@testset "Offline runner" begin
    ctx = Context.load_from_string(raw"""
    @Variable foo -> karabo"camera.data"
    @Variable function bar(x -> foo)
        return x * 2
    end
    @Variable function baz(x -> foo)
        return x + 100
    end
    @Variable function qux(x -> foo)
        @add_subvariable("half", x / 2)
        return x
    end
    """)
    # `values` is keyed by external dependency name; a DimArray with a trainId
    # dimension feeds a distinct element per train.
    vals = Dict("camera.data" => Context.DD.DimArray([1, 2, 3, 4],
                                                     (Context.DD.Dim{:trainId}([10, 11, 12, 13]),)))

    # Filtering helpers
    @test occursin(Context.pattern_regex("baz.*"), "baz.quux")
    @test !occursin(Context.pattern_regex("baz"), "baz.quux")
    @test Context.upstream_closure(ctx.dag, ["bar"]) == Set(["bar", "foo"])

    # `select` keeps the named variable plus its upstream closure, prunes the rest
    r = Context.run(ctx, vals; select=["bar"])
    @test keyset(r) == Set(["foo", "bar"])
    @test collect(r["foo"]) == [1, 2, 3, 4]
    @test collect(r["bar"]) == [2, 4, 6, 8]
    @test collect(Context.DD.lookup(r["foo"], :trainId)) == [10, 11, 12, 13]

    # A subvariable can be selected by name, keeping its parent
    r = Context.run(ctx, vals; select=["qux.half"])
    @test keyset(r) == Set(["foo", "qux", "qux.half"])
    @test collect(r["qux.half"]) == [0.5, 1.0, 1.5, 2.0]

    # A constant value is sent unchanged on every train
    r = Context.run(ctx, Dict("camera.data" => vals["camera.data"], "unused" => 7); select=["bar"])
    @test collect(r["bar"]) == [2, 4, 6, 8]

    # A constant override replaces a variable's output on every train
    r = Context.run(ctx, vals; select=["bar", "baz"], override=Dict("baz" => 1000))
    @test keyset(r) == Set(["foo", "bar", "baz"])
    @test collect(r["baz"]) == fill(1000, 4)

    # A DimArray override carrying a trainId dimension is applied per train, and
    # provides the train clock even when the input upstream is pruned away.
    bg = Context.DD.DimArray([100, 200, 300, 400], (Context.DD.Dim{:trainId}([10, 11, 12, 13]),))
    r = Context.run(ctx, vals; select=["baz"], override=Dict("baz" => bg))
    @test collect(r["baz"]) == [100, 200, 300, 400]

    # The matched trains are the intersection across per-train sources
    bg2 = Context.DD.DimArray([100, 200], (Context.DD.Dim{:trainId}([11, 12]),))
    r = Context.run(ctx, vals; select=["bar", "baz"], override=Dict("baz" => bg2))
    @test collect(Context.DD.lookup(r["bar"], :trainId)) == [11, 12]
    @test collect(r["bar"]) == [4, 6]

    # Overriding a variable cuts its deps, so its unused upstream is pruned
    r = Context.run(ctx, vals; select=["baz"], override=Dict("baz" => 0))
    @test keyset(r) == Set(["baz"])

    # The loaded context is never mutated by a run
    @test keyset(ctx.dag) == Set(["foo", "bar", "baz", "qux"])

    @test_throws XfaContextException Context.run(ctx, vals; override=Dict("nope" => 1))
    @test_throws XfaContextException Context.run(ctx, vals; select=["nomatch"])

    # Nothing carrying a trainId dimension would be an infinite stream
    @test_throws ArgumentError Context.run(ctx, Dict("camera.data" => 5); select=["bar"])

    @testset "DataCollection method" begin
        # Drive the PythonCall extension's run against a real extra-data
        # DataCollection. The AGIPD1M example run carries a constant control
        # property (integrationTime = 15) for 100 trains starting at 10000.
        ctx = Context.load_from_string(raw"""
        @Variable itime -> karabo"SPB_IRU_AGIPD1M1/MDL/FPGA_COMP.integrationTime"

        @Variable function double(x -> itime)
            return x * 2
        end

        @Variable function module0(x -> karabo"SPB_DET_AGIPD1M-1/DET/0CH0:xtdf[image.data]")
            nanmean(x; dim=(:trainId, :dim_0))
        end

        @Variable itime_proxied -> karabo"SPB_IRU_AGIPD1M1/MDL/FPGA_COMP.integrationTime@proxy:output"
        """)

        mktempdir() do dir
            dc = PythonCall.GIL.@lock begin
                pyimport("extra_data.tests.make_examples").make_agipd1m_run(dir)
                pyimport("extra_data").RunDirectory(dir)
            end

            # The type check rejects a Py object that isn't a DataCollection
            @test_throws ArgumentError Context.run(ctx, PythonCall.GIL.@lock(pylist([1, 2, 3])))

            # Test returning scalars
            r = Context.run(ctx, dc; select=["double"])
            @test keyset(r) == Set(["itime", "double"])
            @test length(r["itime"]) == 100
            @test all(==(15), collect(r["itime"]))
            @test all(==(30), collect(r["double"]))
            @test collect(Context.DD.lookup(r["itime"], :trainId))[1:3] == [10000, 10001, 10002]

            # Test returning arrays
            r = Context.run(ctx, dc; select=["module0"])
            @test keyset(r) == Set(["module0"])
            @test size(r["module0"]) == (128, 512, 100)
            @test Context.DD.hasdim(r["module0"], :trainId)
            @test collect(Context.DD.lookup(r["module0"], :trainId))[1:3] == [10000, 10001, 10002]

            # Test that proxied dependencies are ignored
            r = Context.run(ctx, dc; select=["itime_proxied"])
            @test keyset(r) == Set(["itime_proxied"])
            @test all(==(15), collect(r["itime_proxied"]))
        end
    end
end

end
