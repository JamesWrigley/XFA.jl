module XfaTests

__revise_mode__ = :eval

# using Mmap
# using Sockets

using XFA
using ReTest
# using EllipsisNotation
# using InterProcessCommunication: SharedMemory, shmid

using ImGuiTestEngine
import ImGuiTestEngine as te
import CImGui as ig
using XFA.XfaEngine.Context: Dependency, DepKind_Karabo, DepKind_Variable, DepKind_Subvariable,
    karabo_dependency, @karabo_str, SourceInfo
using XfaContext: XfaContext as Context

# Set up the backend for CImGui
import GLFW
import ModernGL
ig.set_backend(:GlfwOpenGL3)

@testset "Settings" begin
    @testset "save_section only overwrites its own section" begin
        config_dir = mktempdir()
        withenv("XFA_CONFIG_DIR" => config_dir) do
            XFA.save_section("SectionA", Dict("key1" => "value1"))
            XFA.save_section("SectionB", Dict("key2" => "value2"))

            settings = XFA.load_settings()
            @test settings["SectionA"]["key1"] == "value1"
            @test settings["SectionB"]["key2"] == "value2"

            # Overwrite SectionA, SectionB should be untouched
            XFA.save_section("SectionA", Dict("key1" => "updated"))
            settings = XFA.load_settings()
            @test settings["SectionA"]["key1"] == "updated"
            @test settings["SectionB"]["key2"] == "value2"
        end
    end

    @testset "GuiState constructor reads contexts from sections" begin
        settings = Dict(
            "GuiState" => Dict(
                "address" => "testhost",
                "engine_environment" => "@test-env",
                "client_type" => 1,
            ),
            "ClientState" => Dict(
                "contexts" => Dict(
                    "/path/to/ctx.jl" => Dict(
                        "plots" => [
                            Dict("type" => "Plot", "name" => "var1", "id" => "var1##plot-1"),
                            Dict("type" => "CorrelationPlot", "id" => "CorrelationPlot##plot-2"),
                        ],
                        "plot_counter" => 5,
                        "saved_layout" => "[Window][Main]\nPos=0,0\n",
                    ),
                ),
            ),
        )

        gui = XFA.GuiState(settings)
        @test gui.address == "testhost"
        @test gui.engine_environment == "@test-env"
        @test haskey(gui.saved_contexts, "/path/to/ctx.jl")
        @test length(gui.saved_contexts["/path/to/ctx.jl"]["plots"]) == 2
        @test isempty(gui.client.plots)
    end

    @testset "GuiState constructor with empty settings uses defaults" begin
        gui = XFA.GuiState(Dict{String,Any}())
        @test isempty(gui.saved_contexts)
    end

    # @testset "restore_plots looks up by context path" begin
    #     gui = XFA.GuiState(Dict(
    #         "ClientState" => Dict("contexts" => Dict(
    #             "ctx_a.jl" => Dict(
    #                 "plot_counter" => 3,
    #                 "saved_layout" => "",
    #                 "plots" => [
    #                     Dict("type" => "Plot", "name" => "x", "id" => "x##plot-1"),
    #                     Dict("type" => "CorrelationPlot", "id" => "CorrelationPlot##plot-2"),
    #                 ],
    #             ),
    #             "ctx_b.jl" => Dict(
    #                 "plot_counter" => 1,
    #                 "saved_layout" => "",
    #                 "plots" => [
    #                     Dict("type" => "Plot", "name" => "y", "id" => "y##plot-1"),
    #                 ],
    #             ),
    #         )),
    #     ))

    #     # Restore context A
    #     gui.client.context_path = "ctx_a.jl"
    #     XFA.restore_plots(gui)

    #     @test length(gui.client.plots) == 2
    #     @test gui.client.plots[1].id == "x##plot-1"
    #     @test gui.client.plots[2].id == "CorrelationPlot##plot-2"
    #     @test gui.plot_counter == 3

    #     # Switch to context B
    #     gui.client.context_path = "ctx_b.jl"
    #     XFA.restore_plots(gui)

    #     @test length(gui.client.plots) == 1
    #     @test gui.client.plots[1].id == "y##plot-1"
    #     @test gui.plot_counter == 1
    # end

    # @testset "restore_plots with unknown context is a no-op" begin
    #     gui = XFA.GuiState(Dict{String,Any}())
    #     gui.client.context_path = "unknown.jl"
    #     XFA.restore_plots(gui)
    #     @test isempty(gui.client.plots)
    # end
end

@testset "Context editing" begin
    @testset "Change argument source" begin
        # Shorthand variable
        source = """
        @Variable foo -> karabo"A/B.prop"
        """
        @test XFA.replace_dep(source, "foo", "data", karabo"C/D.prop") == """
        @Variable foo -> karabo"C/D.prop"
        """

        # Function variable
        source = """
        @Variable function bar(x -> karabo"A/B.prop")
            return x
        end
        """
        @test XFA.replace_dep(source, "bar", "x", karabo"C/D.prop") == """
        @Variable function bar(x -> karabo"C/D.prop")
            return x
        end
        """

        # Only affects the targeted variable
        source = """
        @Variable foo -> karabo"A/B.prop"
        @Variable bar -> karabo"A/B.prop"
        """
        @test XFA.replace_dep(source, "foo", "data", karabo"C/D.prop") == """
        @Variable foo -> karabo"C/D.prop"
        @Variable bar -> karabo"A/B.prop"
        """

        # Unknown variable returns source unchanged
        source = """
        @Variable foo -> karabo"A/B.prop"
        """
        @test_logs (:warn, r"Could not find @Variable") begin
            @test XFA.replace_dep(source, "nonexistent", "data", karabo"C/D.prop") == source
        end

        # Rename fast data sources
        source = """
        @Variable foo -> karabo"A/B:output[x]"
        """
        @test XFA.replace_dep(source, "foo", "data", karabo"C/D:daqOutput[y]") == """
        @Variable foo -> karabo"C/D:daqOutput[y]"
        """

        # Only affects the targeted argument
        source = """
        @Variable function energy(energy -> karabo"SA2_XTD1_XGM/XGM/DOOCS:output[data.intensityTD]",
                                  flux -> karabo"SA2_XTD1_XGM/XGM/DOOCS.pulseEnergy.photonFlux")
            1
        end
        """
        @test XFA.replace_dep(source, "energy", "energy", karabo"foo.bar") == """
        @Variable function energy(energy -> karabo"foo.bar",
                                  flux -> karabo"SA2_XTD1_XGM/XGM/DOOCS.pulseEnergy.photonFlux")
            1
        end
        """

        # Replace karabo"..." with a topic macro
        source = """
        @Variable foo -> karabo"A/B.prop"
        """
        @test XFA.replace_dep(source, "foo", "data", karabo"MID//C/D.prop") == """
        @Variable foo -> karabo"MID//C/D.prop"
        """

        # Replace a topic macro with a different topic macro
        source = """
        @Variable foo -> karabo"MID//A/B.prop"
        """
        @test XFA.replace_dep(source, "foo", "data", karabo"SA2//C/D.newprop") == """
        @Variable foo -> karabo"SA2//C/D.newprop"
        """

        # Karabo → Variable, Variable → Karabo, Subvariable → Karabo
        source = """
        @Variable function baz(x -> karabo"A/B.prop", y -> other_var, z -> foo.half)
            1
        end
        """
        @test XFA.replace_dep(source, "baz", "x", Dependency("my_var")) == """
        @Variable function baz(x -> my_var, y -> other_var, z -> foo.half)
            1
        end
        """
        @test XFA.replace_dep(source, "baz", "y", karabo"C/D.prop") == """
        @Variable function baz(x -> karabo"A/B.prop", y -> karabo"C/D.prop", z -> foo.half)
            1
        end
        """
        @test XFA.replace_dep(source, "baz", "z", karabo"E/F.prop") == """
        @Variable function baz(x -> karabo"A/B.prop", y -> other_var, z -> karabo"E/F.prop")
            1
        end
        """

        # Variable-reference shorthand with an inline argument remap: only the
        # inner arg's karabo dep should change, not the entire RHS.
        source = """
        @Variable jf2 -> jf1(data -> karabo"A/B:daqOutput[data.adc]@A/CAL:dataOutput")
        """
        @test XFA.replace_dep(source, "jf2", "data", karabo"C/D:daqOutput[data.adc]@C/CAL:dataOutput") == """
        @Variable jf2 -> jf1(data -> karabo"C/D:daqOutput[data.adc]@C/CAL:dataOutput")
        """
    end

    @testset "Change group dependency" begin
        source = """
        my_group = MyGroup(; x=karabo"A/B.prop", y=Dependency("energy"))
        """

        # Karabo → different Karabo
        @test XFA.replace_dep(source, "my_group", "x", karabo"E/F.prop") == """
        my_group = MyGroup(; x=karabo"E/F.prop", y=Dependency("energy"))
        """

        # Variable → Karabo
        @test XFA.replace_dep(source, "my_group", "y", karabo"C/D.prop") == """
        my_group = MyGroup(; x=karabo"A/B.prop", y=karabo"C/D.prop")
        """

        # Karabo → Variable
        @test XFA.replace_dep(source, "my_group", "x", Dependency("my_var")) == """
        my_group = MyGroup(; x=Dependency("my_var"), y=Dependency("energy"))
        """

        # Only affects the targeted group instance
        source = """
        g1 = MyGroup(; x=karabo"A/B.prop")
        g2 = MyGroup(; x=karabo"A/B.prop")
        """
        @test XFA.replace_dep(source, "g1", "x", karabo"E/F.prop") == """
        g1 = MyGroup(; x=karabo"E/F.prop")
        g2 = MyGroup(; x=karabo"A/B.prop")
        """

        # Add a new kwarg to a group with no kwargs
        source = """
        my_group = MyGroup()
        """
        @test XFA.replace_dep(source, "my_group", "x", karabo"A/B.prop") == """
        my_group = MyGroup(; x=karabo"A/B.prop")
        """

        # Unknown group returns source unchanged
        @test_logs (:warn, r"Could not find") begin
            @test XFA.replace_dep(source, "nonexistent", "x", karabo"C/D.prop") == source
        end
    end

    @testset "Rename variable" begin
        # Rename shorthand variable
        source = """
        @Variable foo -> karabo"A/B.prop"
        """
        @test XFA.replace_variable_name(source, "foo", "baz") == """
        @Variable baz -> karabo"A/B.prop"
        """

        # Rename function variable
        source = """
        @Variable function foo(x -> karabo"A/B.prop")
            return x
        end
        """
        @test XFA.replace_variable_name(source, "foo", "baz") == """
        @Variable function baz(x -> karabo"A/B.prop")
            return x
        end
        """

        # Rename updates references in other variables
        source = """
        @Variable foo -> karabo"A/B.prop"
        @Variable function bar(x -> foo)
            return x
        end
        """
        @test XFA.replace_variable_name(source, "foo", "baz") == """
        @Variable baz -> karabo"A/B.prop"
        @Variable function bar(x -> baz)
            return x
        end
        """

        # Rename does not affect unrelated variables
        source = """
        @Variable foo -> karabo"A/B.prop"
        @Variable bar -> karabo"C/D.prop"
        """
        @test XFA.replace_variable_name(source, "foo", "baz") == """
        @Variable baz -> karabo"A/B.prop"
        @Variable bar -> karabo"C/D.prop"
        """

        # A group/input is defined by assignment, not a @Variable, and its
        # references inside variables are renamed along with it
        source = """
        my_group = MyGroup(; x=karabo"A/B.prop")
        @Variable function bar(y -> my_group.thing)
            return y
        end
        """
        @test XFA.replace_variable_name(source, "my_group", "baz") == """
        baz = MyGroup(; x=karabo"A/B.prop")
        @Variable function bar(y -> baz.thing)
            return y
        end
        """

        # Unknown names still warn and change nothing
        @test_logs (:warn, r"No occurrences") begin
            @test XFA.replace_variable_name(source, "nonexistent", "baz") == source
        end
    end

    @testset "Set bridge address" begin
        # Add address to a KaraboBridge with no existing address kwarg
        source = """
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("TOPIC", "name"))
        """
        @test XFA.replace_constructor_kwarg(source, "bridge", "address", "\"tcp://foo:1234\"") == """
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("TOPIC", "name"), address="tcp://foo:1234")
        """

        # Replace existing address kwarg
        source = """
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("TOPIC", "name"), address="tcp://old:5555")
        """
        @test XFA.replace_constructor_kwarg(source, "bridge", "address", "\"tcp://new:1234\"") == """
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("TOPIC", "name"), address="tcp://new:1234")
        """

        # Bridge not found returns source unchanged
        source = """
        bridge = KaraboBridge(; trainmatcher=KaraboDevice("TOPIC", "name"))
        """
        @test_logs (:warn, r"Could not find constructor.*") begin
            @test XFA.replace_constructor_kwarg(source, "other", "address", "\"tcp://foo:1234\"") == source
        end
    end

    @testset "Edit RectROI parameter" begin
        roi = XFA.XfaEngine.Context.RectROI(1.5, 2.5, 10.0, 20.0)
        @test XFA.format_param_value(roi) == "RectROI(1.5, 2.5, 10.0, 20.0)"

        # Top-level `roi = Parameter(RectROI())` assignment
        source = """
        roi = Parameter(RectROI())
        """
        @test XFA.replace_parameter_value(source, "roi", XFA.format_param_value(roi)) == """
        roi = Parameter(RectROI(1.5, 2.5, 10.0, 20.0))
        """

        # Replaces the existing value, preserving the Parameter wrapper
        source = """
        roi = Parameter(RectROI(0.0, 0.0, 1.0, 1.0))
        """
        @test XFA.replace_parameter_value(source, "roi", XFA.format_param_value(roi)) == """
        roi = Parameter(RectROI(1.5, 2.5, 10.0, 20.0))
        """

        # Inside a group constructor as a kwarg
        source = """
        my_group = MyGroup(; roi=RectROI())
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "roi",
                                            XFA.format_param_value(roi)) == """
        my_group = MyGroup(; roi=RectROI(1.5, 2.5, 10.0, 20.0))
        """

        # Missing assignment returns source unchanged with a warning
        source = """
        other = Parameter(RectROI())
        """
        @test_logs (:warn, r"Could not find Parameter assignment.*") begin
            @test XFA.replace_parameter_value(source, "roi", "RectROI(1.0, 2.0, 3.0, 4.0)") == source
        end

        @test XFA.format_param_value(Context.LinearROI(1.5, 10.0; axis=:y)) == "LinearROI(1.5, 10.0; axis=:y)"
    end

    @testset "Edit Int parameter" begin
        @test XFA.format_param_value(42) == "42"

        source = """
        count = Parameter(0)
        """
        @test XFA.replace_parameter_value(source, "count", XFA.format_param_value(42)) == """
        count = Parameter(42)
        """

        source = """
        my_group = MyGroup(; count=3)
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "count",
                                            XFA.format_param_value(42)) == """
        my_group = MyGroup(; count=42)
        """
    end

    @testset "Edit vector parameter" begin
        @test XFA.format_param_value([1, 2, 3]) == "[1, 2, 3]"
        @test XFA.format_param_value([1.5, 2.0]) == "[1.5, 2.0]"
        @test XFA.format_param_value(Int[]) == "[]"

        source = """
        bins = Parameter([0, 1])
        """
        @test XFA.replace_parameter_value(source, "bins", XFA.format_param_value([1, 2, 3])) == """
        bins = Parameter([1, 2, 3])
        """

        source = """
        my_group = MyGroup(; bins=[0.0])
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "bins",
                                            XFA.format_param_value([1.5, 2.0])) == """
        my_group = MyGroup(; bins=[1.5, 2.0])
        """
    end

    @testset "Edit string group parameter" begin
        # Replace an existing string kwarg on a generic group constructor
        source = """
        my_group = MyGroup(; label="old", count=3)
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "label", "\"new\"") == """
        my_group = MyGroup(; label="new", count=3)
        """

        # Append the kwarg when the group has no parameter section yet
        source = """
        my_group = MyGroup()
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "label", "\"hello\"") == """
        my_group = MyGroup(; label="hello")
        """

        # Strings with quotes and backslashes round-trip through escape_string
        new_value = "with \"quotes\" and \\ slash"
        source = """
        my_group = MyGroup(; label="plain")
        """
        @test XFA.replace_constructor_kwarg(source, "my_group", "label",
                                            "\"$(escape_string(new_value))\"") == """
        my_group = MyGroup(; label="with \\"quotes\\" and \\\\ slash")
        """
    end

    @testset "Add variable" begin
        variable_spec(origin) = Context.VariableSpec("scan", Context.VariableKind_Variable, origin,
                                                     Context.OrderedDict{String, Dependency}(),
                                                     String[], String[],
                                                     Context.OrderedDict{Symbol, Context.Parameter}())
        group_spec(origin, params) = Context.VariableSpec("MyGroup", Context.VariableKind_Group,
                                                          origin,
                                                          Context.OrderedDict{String, Dependency}(),
                                                          String[], String[], params)
        # Adds to whatever `source` currently holds
        add(spec, name; deps=Context.OrderedDict{String, Dependency}(),
            params=Context.OrderedDict{Symbol, Context.Parameter}()) =
            XFA.add_variable_source(source, spec, name, deps, params)

        source = """
        @Variable foo -> karabo"A/B.prop"
        """

        # A variable's wired deps become overrides on the origin reference; Karabo,
        # variable and subvariable deps each get their own source form.
        deps = Context.OrderedDict("x" => karabo"C/D.prop",
                                   "y" => Dependency("foo"),
                                   "z" => Context.subvariable_dependency("foo", "mean"))
        @test add(variable_spec("XfaContext.scan"), "my_scan"; deps) == """
        @Variable foo -> karabo"A/B.prop"

        @Variable my_scan -> XfaContext.scan(x -> karabo"C/D.prop", y -> foo, z -> foo.mean)
        """

        # Unwired and group deps are left off, so a bare reference is emitted
        deps = Context.OrderedDict("x" => Dependency(""),
                                   "group" => Context.group_dependency(Context.RectROI))
        @test add(variable_spec("XfaContext.scan"), "my_scan"; deps) == """
        @Variable foo -> karabo"A/B.prop"

        @Variable my_scan -> XfaContext.scan
        """

        # An origin from a module that isn't in scope gets an import, inserted
        # after the last existing one. XfaContext is always in scope, so it doesn't.
        source = """
        using XfaContext: Dependency
        import Foo

        @Variable foo -> karabo"A/B.prop"
        """
        @test add(variable_spec("XfaEngine.scan"), "my_scan") == """
        using XfaContext: Dependency
        import Foo
        using XfaEngine: XfaEngine

        @Variable foo -> karabo"A/B.prop"

        @Variable my_scan -> XfaEngine.scan
        """
        @test add(variable_spec("XfaContext.scan"), "my_scan") == """
        using XfaContext: Dependency
        import Foo

        @Variable foo -> karabo"A/B.prop"

        @Variable my_scan -> XfaContext.scan
        """

        # `using X: X`, `import X` and `using X` bind X, `using X: name` doesn't
        for stmt in ("using Foo: Foo", "import Foo", "using Foo")
            @test XFA.ensure_import("$(stmt)\n", "Foo.bar") == "$(stmt)\n"
        end
        @test XFA.ensure_import("using Foo: bar\n", "Foo.bar") == "using Foo: bar\nusing Foo: Foo\n"
        @test XFA.ensure_import("x = 1\n", "Foo.bar") == "using Foo: Foo\n\nx = 1\n"

        # Groups become a constructor call; only wired deps and parameters edited
        # away from the spec's default are passed as kwargs.
        params = Context.OrderedDict(:threshold => Context.Parameter(800.0),
                                     :label => Context.Parameter("old"))
        spec = group_spec("MyLib.MyGroup", params)
        source = """
        @Variable foo -> karabo"A/B.prop"
        """
        @test add(spec, "my_group";
                  deps=Context.OrderedDict("x" => karabo"C/D.prop", "y" => Dependency("foo")),
                  params=Context.OrderedDict(:threshold => Context.Parameter(900.0),
                                             :label => Context.Parameter("old"))) == """
        using MyLib: MyLib

        @Variable foo -> karabo"A/B.prop"

        my_group = MyLib.MyGroup(; x=karabo"C/D.prop", y=Dependency("foo"), threshold=900.0)
        """

        # A group with nothing wired or edited falls back to the constructor defaults
        @test add(spec, "my_group"; params=deepcopy(params)) == """
        using MyLib: MyLib

        @Variable foo -> karabo"A/B.prop"

        my_group = MyLib.MyGroup()
        """

        # An edited parameter whose value has no source representation is an error
        unwritable = Context.OrderedDict(:threshold => Context.Parameter(800.0),
                                         :label => Context.Parameter(Context.KaraboDevice("MID", "A/B/C")))
        @test_throws "Cannot represent parameter 'label'" add(spec, "my_group"; params=unwritable)
    end

    @testset "Remove variable" begin
        # Every definition form is removed whole, and the surviving neighbours
        # keep a blank line between them
        source = """
        my_group = MyGroup(; x=karabo"A/B.prop")

        @Variable function foo(x -> my_group.thing)
            return x
        end

        @Variable bar -> karabo"C/D.prop"
        """
        @test XFA.remove_variable_source(source, "foo") == """
        my_group = MyGroup(; x=karabo"A/B.prop")

        @Variable bar -> karabo"C/D.prop"
        """
        @test XFA.remove_variable_source(source, "my_group") == """
        @Variable function foo(x -> my_group.thing)
            return x
        end

        @Variable bar -> karabo"C/D.prop"
        """
        @test XFA.remove_variable_source(source, "bar") == """
        my_group = MyGroup(; x=karabo"A/B.prop")

        @Variable function foo(x -> my_group.thing)
            return x
        end
        """

        # Removing the only declaration empties the file
        @test XFA.remove_variable_source("@Variable bar -> karabo\"C/D.prop\"\n", "bar") == ""

        @test_throws "Could not find a definition for 'nonexistent'" XFA.remove_variable_source(source, "nonexistent")
    end
end

@testset "Dependency completions" begin
    SI(topic, name, ambiguous) = SourceInfo(topic, name, "", ambiguous)
    source_list = [SI("MID", "MID_DET/CAM/1", true),
                   SI("MID", "MID_EXP/MOTOR/1", false),
                   SI("SA2", "SA2_XTD1_XGM/XGM/DOOCS", false),
                   SI("SA2", "MID_DET/CAM/1", true)]

    # A device is inserted bare when its name is unique, and topic-qualified
    # when the same name exists under more than one topic.
    @test XFA.source_base(SI("MID", "MID_EXP/MOTOR/1", false)) == "MID_EXP/MOTOR/1"
    @test XFA.source_base(SI("MID", "MID_DET/CAM/1", true)) == "MID//MID_DET/CAM/1"
    @test XFA.source_base(SI("SA2", "MID_DET/CAM/1", true)) == "SA2//MID_DET/CAM/1"

    # A "TOPIC//" prefix splits off as the fixed topic, leaving the fuzzy query
    @test XFA.split_topic("MID//DET") == ("MID", "DET")
    @test XFA.split_topic("MID_EXP") == ("", "MID_EXP")

    # The topic prefix restricts the device list to that topic
    topic_fixed, query = XFA.split_topic("MID//DET")
    items = filter(s -> s.topic == topic_fixed, source_list)
    @test all(s -> s.topic == "MID", items)
    # The substring match outranks the (fuzzy) MID_EXP/MOTOR/1 d-e-t match
    @test first(XFA.fuzzy_match(query, items, s -> s.name))[2].name == "MID_DET/CAM/1"

    # A channel suffix isn't part of the device being completed
    @test XFA.strip_channel("MID_DET/CAM/1:daqOutput") == "MID_DET/CAM/1"
    @test XFA.source_channel("MID_DET/CAM/1:daqOutput") == "daqOutput"
    @test XFA.strip_channel("MID_EXP/MOTOR/1") == "MID_EXP/MOTOR/1"
    @test XFA.source_channel("MID_EXP/MOTOR/1") == ""

    # Property completions come from the cached device schema
    props(names) = XFA.PropertyList(names, String[], String[], String[])
    client = XFA.ClientState()
    client.source_list = source_list
    client.source_properties[("MID", "MID_EXP/MOTOR/1")] =
        XFA.DeviceProperties(props(["pos", "velocity"]), Dict("daqOutput" => props(["data.image"])))

    # A bare device completes against its slow properties, but only where slow
    # sources are allowed; a device:channel completes against its fast data paths
    @test XFA.property_completions(client, "MID_EXP/MOTOR/1", true) == ["pos", "velocity"]
    @test XFA.property_completions(client, "MID_EXP/MOTOR/1", false) == String[]
    @test XFA.property_completions(client, "MID_EXP/MOTOR/1:daqOutput", false) == ["data.image"]

    # Unknown devices and channels complete to nothing
    @test XFA.property_completions(client, "MID_EXP/MOTOR/1:nope", true) == String[]
    @test XFA.property_completions(client, "GHOST/DEV/1", true) == String[]
end

@testset "remap_source" begin
    client = XFA.ClientState()
    client.source_list = [SourceInfo("MID", "camera", "AravisBaslerCamera"),
                          SourceInfo("MID", "motor", "Motor")]

    camera_rule = XFA.RemapRule(XFA.RemapKind_Simple,
                                raw"^(.*):output\[data\.image\.pixels\]$",
                                "AravisBaslerCamera",
                                raw"\1:output[data.image.data]")
    client.remap_rules = [camera_rule]

    # Matching device class + matching source: rewritten
    @test XFA.remap_source(client, "camera:output[data.image.pixels]", Ref{Any}(nothing)) ==
          ("camera:output[data.image.data]", nothing)

    # Wrong device class: pass through
    @test XFA.remap_source(client, "motor:output[data.image.pixels]", Ref{Any}(nothing)) ==
          ("motor:output[data.image.pixels]", nothing)

    # Source regex doesn't match: pass through
    @test XFA.remap_source(client, "camera.something", Ref{Any}(nothing)) == ("camera.something", nothing)

    # Unknown device (no classId resolved): camera_rule's class regex still
    # only matches "AravisBaslerCamera", so pass through.
    @test XFA.remap_source(client, "ghost:output[data.image.pixels]", Ref{Any}(nothing)) ==
          ("ghost:output[data.image.pixels]", nothing)

    # All matching rules apply in order, cumulatively
    push!(client.remap_rules,
          XFA.RemapRule(XFA.RemapKind_Simple, raw"data\.image\.data", "", "data.image.foo"))
    @test XFA.remap_source(client, "camera:output[data.image.pixels]", Ref{Any}(nothing)) ==
          ("camera:output[data.image.foo]", nothing)

    # Empty source + empty device_class is rejected — it would match every
    # source unconditionally, which is almost certainly a config mistake.
    @test_throws ArgumentError XFA.RemapRule(XFA.RemapKind_Simple, "", "", "REPLACED")
end

@testset "sampled_pctile!" begin
    buf = Int32[]
    # True 1st/99th percentiles of the valid subset, for the approximate
    # histogram estimate to be checked against.
    pct(v) = (XFA.nanpctile(v, 1), XFA.nanpctile(v, 99))

    # Histogram estimate lands within a bin's width of the true percentiles.
    # Small array (<1000) uses stride=1.
    small = reshape(collect(1.0:100.0), 10, 10)
    e1, e99 = pct(vec(small))
    p1, p99 = XFA.sampled_pctile!(buf, small)
    @test p1 ≈ e1 atol = 1.0
    @test p99 ≈ e99 atol = 1.0

    # Constant matrix is exact (degenerate range), and exercises the strided
    # path for a large (>=1000) input.
    @test XFA.sampled_pctile!(buf, fill(7.0, 100, 100)) == (7.0, 7.0)

    # Non-finite samples (NaN, Inf, -Inf) are dropped.
    mixed = [1.0 NaN Inf; 2.0 -Inf 99.0; 25.0 NaN 100.0]
    e1, e99 = pct(filter(isfinite, vec(mixed)))
    p1, p99 = XFA.sampled_pctile!(buf, mixed)
    @test isfinite(p1) && isfinite(p99)
    @test p1 ≈ e1 atol = 1.0
    @test p99 ≈ e99 atol = 1.0

    # All non-finite / empty → fallback, still finite.
    @test XFA.sampled_pctile!(buf, [NaN Inf; -Inf NaN]) == (0.0, 1.0)
    @test XFA.sampled_pctile!(buf, Matrix{Float64}(undef, 0, 0)) == (0.0, 1.0)

    # Integer input filters nothing (no non-finite ints) but still works.
    ints = reshape(collect(Int32(1):Int32(100)), 10, 10)
    e1, e99 = pct(vec(ints))
    p1, p99 = XFA.sampled_pctile!(buf, ints)
    @test p1 ≈ e1 atol = 1.0
    @test p99 ≈ e99 atol = 1.0

    # Log mode: log10 of the positive-only percentiles.
    small_pos = reshape(10.0 .^ collect(0.0:0.01:0.99), 10, 10)
    e1, e99 = pct(vec(small_pos))
    p1, p99 = XFA.sampled_pctile!(buf, small_pos, true)
    @test p1 ≈ log10(e1) atol = 0.05
    @test p99 ≈ log10(e99) atol = 0.05

    # Non-positive samples are dropped before log10.
    mixed_log = [-1.0 0.0 NaN; 1.0 10.0 NaN; 100.0 1000.0 NaN]
    e1, e99 = pct(filter(x -> isfinite(x) && x > 0, vec(mixed_log)))
    p1, p99 = XFA.sampled_pctile!(buf, mixed_log, true)
    @test p1 ≈ log10(e1) atol = 0.1
    @test p99 ≈ log10(e99) atol = 0.1

    # No positive samples → fallback, never NaN/-Inf from log10(≤0).
    @test XFA.sampled_pctile!(buf, [-1.0 0.0; 0.0 -2.0], true) == (0.0, 1.0)
end

@testset "Fitting" begin
    @testset "fit_gaussian" begin
        # Recover known parameters from clean data.
        y0, A, μ, σ = 1.0, 5.0, 2.0, 0.5
        x = collect(range(-2.0, 6.0; length=200))
        y = @. XFA.gaussian(x, y0, A, μ, σ)
        popt, retcode = XFA.fit_gaussian(y, x)
        @test popt ≈ [y0, A, μ, σ] atol=1e-6
        @test retcode == :Success

        # Non-finite ydata is masked out; xdata defaults to 1:length(ydata).
        y_noisy = copy(y)
        y_noisy[[10, 50, 100]] .= [NaN, Inf, -Inf]
        @test XFA.fit_gaussian(y_noisy, x)[1] ≈ [y0, A, μ, σ] atol=1e-6

        # No finite samples → nothing popt + :NoFiniteSamples retcode.
        popt, retcode = XFA.fit_gaussian(fill(NaN, 10))
        @test isnothing(popt)
        @test retcode == :NoFiniteSamples

        # A_sign=-1 fits a downward peak.
        y_down = @. XFA.gaussian(x, 0.0, -3.0, 1.0, 0.4)
        popt, _ = XFA.fit_gaussian(y_down, x; A_sign=-1)
        @test popt[2] < 0
        @test popt[3] ≈ 1.0 atol=1e-4

        # Invalid p0 length is rejected.
        @test_throws ArgumentError XFA.fit_gaussian(y, x; p0=[1.0, 2.0])

        # Fixed parameters: pinned slots stay put, free slots converge. Pins
        # exercise the ForwardDiff path through the closure.
        popt, retcode = XFA.fit_gaussian(y, x; fixed=[y0, nothing, nothing, nothing])
        @test popt ≈ [y0, A, μ, σ] atol=1e-6
        @test retcode == :Success
        popt, _ = XFA.fit_gaussian(y, x; fixed=[nothing, nothing, μ, σ])
        @test popt[3] == μ && popt[4] == σ
        @test popt ≈ [y0, A, μ, σ] atol=1e-6
        # All slots pinned → return the pinned vector directly.
        @test XFA.fit_gaussian(y, x; fixed=[y0, A, μ, σ]) == ([y0, A, μ, σ], :Success)
    end

    @testset "fit_erf" begin
        # Recover known parameters from a clean step
        y0, A, center, width = 0.5, 2.0, 1.0, 1.5
        x = collect(range(-3.0, 5.0; length=200))
        y = @. XFA.erf(x, y0, A, center, width)
        popt, retcode = XFA.fit_erf(y, x)
        @test popt ≈ [y0, A, center, width] atol=1e-4
        @test retcode == :Success

        # Non-finite ydata is masked out
        y_noisy = copy(y)
        y_noisy[[10, 50, 100]] .= [NaN, Inf, -Inf]
        @test XFA.fit_erf(y_noisy, x)[1] ≈ [y0, A, center, width] atol=1e-4

        # No finite samples → nothing popt + :NoFiniteSamples retcode.
        popt, retcode = XFA.fit_erf(fill(NaN, 10))
        @test isnothing(popt)
        @test retcode == :NoFiniteSamples

        # Invalid p0 length is rejected.
        @test_throws ArgumentError XFA.fit_erf(y, x; p0=[1.0])
    end

    @testset "fit_sin" begin
        # Multi-cycle data with a non-trivial phase shift.
        y0, A, period, φ = 0.5, 2.0, 3.0, 0.4
        x = collect(range(0.0, 30.0; length=600))
        y = @. XFA.sinusoid(x, y0, A, period, φ)
        popt, retcode = XFA.fit_sin(y, x)
        @test popt ≈ [y0, A, period, φ] atol=1e-4
        @test retcode == :Success

        # Invalid p0 length is rejected.
        @test_throws ArgumentError XFA.fit_sin(y, x; p0=[1.0, 2.0])

        # No finite samples → nothing popt + :NoFiniteSamples retcode.
        @test XFA.fit_sin(fill(NaN, 10)) == (nothing, :NoFiniteSamples)
    end

    @testset "fit_line" begin
        line(x, intercept, slope) = intercept + slope * x

        intercept = 0.5
        slope = 2.0
        x = collect(range(-3.0, 5.0; length=50))
        y = @. line(x, intercept, slope)
        popt, retcode = XFA.fit_line(y, x)
        @test popt ≈ [slope, intercept] atol=1e-10
        @test retcode == :Success

        # Non-finite ydata is masked out.
        y_noisy = copy(y)
        y_noisy[[5, 25]] .= [NaN, Inf]
        @test XFA.fit_line(y_noisy, x)[1] ≈ [slope, intercept] atol=1e-10

        # Failure modes carry the right retcode.
        @test XFA.fit_line(fill(NaN, 10)) == (nothing, :NoFiniteSamples)
        @test XFA.fit_line([1.0]) == (nothing, :InsufficientData)
    end

    @testset "Restrict X" begin
        # Only samples inside the ROI are fitted (X needn't be sorted, as for
        # scatter data) and the model curve spans it.
        x = collect(0.0:0.1:10.0)[[1:2:101; 2:2:100]]
        y = [2.0 <= xi <= 4.0 ? 3.0 * xi + 1.0 : 100.0 for xi in x]
        fit = XFA.FitSettings(; fit_type=Ref(Cint(findfirst(==("Line"), XFA.FIT_TYPES) - 1)),
                              restrict_x=true, x_roi=XFA.LinearROI(2.0, 2.0))
        XFA.reset_fit_params!(fit)
        XFA.compute_fit!(fit, y, x)
        @test fit.popt ≈ [3.0, 1.0] atol=1e-10
        @test extrema(fit.model_x) == (2.0, 4.0)
    end
end

@testset "GUI" begin
    config_dir = mktempdir()
    test_engine = te.CreateContext(; exit_on_completion=true)

    @register_test(test_engine, "XFA", "Settings") do
        SetRef("Main window")
    end

    withenv("XFA_CONFIG_DIR" => config_dir) do
        t, state = XFA.main(; test_engine)
        wait(t)
    end
    te.DestroyContext(test_engine)
end

## Shared memory stuff, should probably be moved to XfaEngine

# const epix = Device("MID_EXP_EPIX-1/DET/RECEIVER",
#                     ":foo" => (
#                         "bar" => Float16[100, 100],
#                         "baz" => Int),
#                     ":daqOutput" => (
#                         "data.image.pixels" => Float32[704, 768],
#                   "data.backTemp" => Float32),
#                     "rxConf" => (
#                         "rxLane" => Int,
#                         "rxVc" => Int,
#                         "save" => Bool),
#                     "relHumidity" => Float32)

# const motor = Device("MID_EXP_UPP/MOTOR/R1",
#                      "actualPosition" => Float32,
#                      "targetPosition" => Float32)

# # Helper function that reads from/writes to an array in shared memory
# function access_shmem(id, dtype, dims)
#     handle = SharedMemory(id)
#     array = unsafe_wrap(Array, reinterpret(Ptr{dtype}, pointer(handle)), dims)

#     # Modify the array
#     array[end] = array[1] * 2

#     return array[1]
# end

# # Helper function to handle setup/teardown. The main purpose of this is to
# # test ShmemHandle, so all other args/kwargs are forwarded to the
# # ShmemHandle constructor.
# function shmem_fixture(f::Function, name="foo", shape=(10, 10), args...; generator=nothing, mkproc=false, kwargs...)
#     if generator == nothing
#         handle = ShmemHandle(name, shape, args...; kwargs...)
#     else
#         handle = ShmemHandle(generator, name, shape, args...; kwargs...)
#     end

#     module_path = dirname(@__DIR__)
#     if mkproc
#         pid = addprocs(1; exeflags="--project=$(module_path)")[1]
#     end

#     try
#         if mkproc
#             f(handle, pid)
#         else
#             f(handle)
#         end
#     finally
#         finalize(handle)
#         if mkproc
#             rmprocs(pid)
#         end
#     end
# end

# @testset "devices.jl" begin
#     @testset "ShmemHandle" begin
#         shmem_fixture() do handle
#             # Test the default pipeline name
#             @test shmid(handle.buffer) == "/foo:dataOutput"

#             # Check that the finalizer actually frees the shared memory
#             finalize(handle)
#             @test_throws SystemError SharedMemory(shmid(handle.buffer))
#         end

#         generator = (trainid, out) -> fill!(out, trainid)
#         test_shape = (128, 512)
#         dtype = Float32
#         num_slots = 2
#         shmem_fixture("foo", test_shape; generator, mkproc=true, output_pipeline="bar", dtype, num_slots) do handle, pid
#             # Check the shared mem ID
#             @test shmid(handle.buffer) == "/foo:bar"

#             # Check the size of the buffer
#             @test sizeof(handle.buffer) == prod(test_shape) * sizeof(dtype) * num_slots

#             # Check that we can open the buffer from another process
#             handle.array[1] = 42
#             @everywhere pid include(@__FILE__)

#             remote_value = remotecall_fetch(access_shmem, pid,
#                                             shmid(handle.buffer), dtype, (test_shape..., num_slots))

#             # Check that that the other process can read from and write to the array
#             @test remote_value == handle.array[1]
#             @test handle.array[end] == handle.array[1] * 2

#             # Test data generation. Note that we hard-code the resulting string
#             # for the sake of clarity on what to expect it to look like, though
#             # the string will need to be updated if the tests are changed.
#             @test nextslot(handle, 1) == raw"/foo:bar$float32$128,512,2$1"
#             @test all(handle.array[.., 1] .== 1)

#             # Go to the next slot
#             @test nextslot(handle, 42) == raw"/foo:bar$float32$128,512,2$2"
#             @test all(handle.array[.., 2] .== 42)

#             # Wrap-around
#             @test nextslot(handle, 314) == raw"/foo:bar$float32$128,512,2$1"
#             @test all(handle.array[.., 1] .== 314)
#         end
#     end

#     @testset "Device" begin
#         test_device = Device("Foo",
#                              "slithy" => "toves",
#                              ":bar" => (
#                                  "baz" => 1,
#                                  "quux" => 2
#                              ))

#         # Test get_control_properties() and get_instrument_sources()
#         @test length(get_control_properties(test_device)) == 1
#         @test length(get_instrument_sources(test_device)) == 1

#         # Test indexing by source and property names
#         @test test_device["slithy"] == "toves"
#         @test test_device[":bar"] isa Dict
#         @test test_device[":bar"]["baz"] == 1
#         @test test_device[":bar", "quux"] == 2

#         @test_throws ErrorException test_device["foo"]
#         @test_throws ErrorException test_device[":bar", "bars"]

#         # Test the Device finalizer
#         shmem_fixture() do handle
#             device_with_shmem = Device("Foo", "bar" => handle)
#             finalize(device_with_shmem)

#             @test_throws SystemError SharedMemory(shmid(handle.buffer))
#         end
#     end

#     @testset "DeviceGroup" begin
#         agipd = makeagipd("MID")
#         try
#             # Test that we generate the right number of modules
#             @test length(agipd) == 16

#             # Test the finalizer
#             finalize(agipd)
#             handle_ids = [shmid(d[":dataOutput", "image.data"].buffer) for d in agipd.devices]
#             for id in handle_ids
#                 @test_throws SystemError SharedMemory(id)
#             end
#         catch e
#             finalize(agipd)
#             rethrow(e)
#         end
#     end
# end

# @testset "sim_onc.jl" begin
#     # Create some devices
#     devices = []
#     push!(devices, epix)

#     agipd = makeagipd("MID")
#     @test length(agipd) == 16
#     push!(devices, agipd)

#     # Find an available port
#     port = getavailableport(42000)

#     # Creating a cluster with the wrong number of bridges should fail
#     @test_throws ArgumentError OnlineCluster(devices, Int[])
#     @test_throws ArgumentError OnlineCluster(devices, [port, port + 1])

#     function sim_onc_fixture(f::Function, devices, ports)
#         onc = OnlineCluster(devices, ports)

#         # Attach a client to the bridge server
#         endpoint = first(keys(onc.servers))
#         client = KaraboBridgeClient(endpoint; timeout=2)

#         try
#             f(onc, client)
#         finally
#             close(client)
#             finalize(onc)
#         end
#     end

#     # Test finalizer of OnlineCluster
#     shmem_fixture() do handle
#         test_device = Device("Foo", "bar" => handle)

#         sim_onc_fixture([test_device], [port]) do onc, client
#             # Sanity check that the buffer is created
#             @test_nowarn SharedMemory(shmid(handle.buffer))
#         end

#         # After the above block ends the OnlineCluster should be finalized,
#         # deleting the buffer.
#         @test_throws SystemError SharedMemory(shmid(handle.buffer))
#     end

#     # Create a mock online cluster with a single trainmatcher
#     sim_onc_fixture(devices, [port]) do onc, client
#         # Start it
#         t = startonc(onc)
#         @test istaskstarted(t)

#         # Stop it and check that it stops within a certain timeout
#         stoponc(onc)
#         @test timedwait(() -> istaskdone(t), 1) == :ok

#         # The number of trains sent depends on the size of the servers buffers, but
#         # it ought to be at least 1.
#         @test onc.sent_trains > 0

#         # Start it again
#         t = startonc(onc)

#         # Get some data
#         data, metadata = next(client)

#         # Check that all the slow data properties are present
#         for device in get_all_devices(devices)
#             control_properties = get_control_properties(device)
#             if !isempty(control_properties)
#                 @test device.name ∈ keys(data)

#                 for prop in control_properties
#                     @test prop ∈ keys(data[device.name])
#                 end
#             end

#             # And the instrument sources
#             instrument_sources = get_instrument_sources(device)
#             for source in instrument_sources
#                 @test source ∈ keys(data)

#                 source_properties = Dict(x for x in device if startswith(x[1], source))
#                 @test length(source_properties) > 0
#                 for (key, prop_type) in pairs(source_properties)
#                     # Check that the property is present
#                     prop_name = split(key, '[')[2][1:end - 1]
#                     @test prop_name ∈ keys(data[source])

#                     # And that arrays have the right shape
#                     if prop_type isa Array && eltype(prop_type) <: Real
#                         # Note: the bridge client currently assumes that all arrays are
#                         # row-major (Python/numpy default), so the axis order will be
#                         # swapped to represent the array as a column-major Julia array.
#                         shape = Tuple(Int.(prop_type))
#                         @test size(data[source][prop_name]) == reverse(shape)
#                     end
#                 end
#             end
#         end

#         # Stop the mock cluster
#         stoponc(onc)
#         @test timedwait(() -> istaskdone(t), 1) == :ok
#     end
# end

end
