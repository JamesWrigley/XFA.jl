using Printf: @sprintf
import Base.ScopedValues: ScopedValue, @with

using CImGui: CImGui as ig, ImVec2, ImVec4, IM_COL32
using CImGui.CSyntax: @c
using ImPlot: ImPlot
using GLFW: GLFW
using ModernGL
import ImGuiNodeEditor as ne

using NaNStatistics: nanpctile
using DimensionalData: DimensionalData as DD, DimVector, DimMatrix, DimArray, At, lookup
using DataStructures: CircularBuffer, OrderedDict
using XfaContext: Parameter, OptionalDims, KaraboDevice, Dependency, karabo_dependency,
    ArrayMetadata, RectROI, PlotSpec, LayerSpec,
    Scalar1dScan, positions
include("plotting.jl")

using LibSSH: LibSSH as ssh
using HTTP: HTTP, WebSockets
using XfaEngine: EngineState, getavailableport, RoutingRule, RemapRule, RemapKind,
    RemapKind_Simple, RemapKind_Proxy
using Dates: Dates, unix2datetime, @dateformat_str
using XfaEngine.ZfpWorkspaces: ZfpWorkspace, CompressedArray, decompress_array,
    decompress_array!, allocate_array, restore_dims
include("states.jl")

using TOML: TOML
using Sockets: Sockets
using CRC32c: crc32c
using Serialization
using XfaEngine.Protocol
using XfaEngine: XfaEngine, Protocol
using XfaContext: Dependency, DependencyKind, DepKind_Variable, DepKind_Subvariable, DepKind_Karabo, DepKind_Group,
    karabo_dependency, karabo_dep_string, Parameter, KaraboDevice, VariableData, ArrayMetadata, OptionalDims

include("imgui_helpers.jl")
include("state_inspector.jl")
include("client.jl")
include("context_edit.jl")
include("variable_widgets.jl")

import Revise

const state = ScopedValue{GuiState}()

node_handle(client, id) = get!(() -> ne.NodeId(id), client.ne_node_handles, id)
pin_handle(client, id) = get!(() -> ne.PinId(id), client.ne_pin_handles, id)
link_handle(client, id) = get!(() -> ne.LinkId(id), client.ne_link_handles, id)

# The node editor persists its own state (node positions, pan, zoom) through
# these callbacks. It hands us a JSON string to stash and reads it back on load;
# save_settings writes ne_settings out to settings.toml.
function ne_save_settings(data::Ptr{Cchar}, size::Csize_t, reason, user::Ptr{Cvoid})::Bool
    client = unsafe_pointer_to_objref(user)::ClientState
    client.ne_settings = unsafe_string(data, size)
    return true
end

# Called first with a null buffer to query the size, then with a buffer to fill.
function ne_load_settings(data::Ptr{Cchar}, user::Ptr{Cvoid})::Csize_t
    client = unsafe_pointer_to_objref(user)::ClientState
    blob = client.ne_settings
    if data != C_NULL && !isempty(blob)
        unsafe_copyto!(Ptr{UInt8}(data), pointer(blob), ncodeunits(blob))
    end
    return Csize_t(ncodeunits(blob))
end

# Create the editor with our save/load callbacks wired up. ne_settings must hold
# the current context's blob first so the editor's initial LoadSettings restores
# the layout. The editor is recreated this way whenever the context changes.
function create_node_editor!(client)
    save_cb = @cfunction(ne_save_settings, Bool, (Ptr{Cchar}, Csize_t, ne.SaveReasonFlags, Ptr{Cvoid}))
    load_cb = @cfunction(ne_load_settings, Csize_t, (Ptr{Cchar}, Ptr{Cvoid}))

    config = ne.Config()
    config.SettingsFile = Ptr{Cchar}(C_NULL)
    config.SaveSettings = save_cb
    config.LoadSettings = load_cb
    config.UserPointer = pointer_from_objref(client)

    # CreateEditor copies the config by value, so the transient one can go.
    client.ne_editor = ne.CreateEditor(config)
    ne.Destroy(config)
    client.ne_editor_path = client.context_path
end

# Draw a filled circle the size of a text line. The node editor has no built-in
# pin shapes, so we draw the marker imnodes used to draw automatically.
function draw_pin_icon()
    sz = ig.GetTextLineHeight()
    p = ig.GetCursorScreenPos()
    ig.AddCircleFilled(ig.GetWindowDrawList(), (p.x + sz * 0.5f0, p.y + sz * 0.5f0),
                       sz * 0.35f0, IM_COL32(204, 204, 204, 255))
    ig.Dummy(sz, sz)
end

# Draw a pin marker. The pivot anchors the link to the left edge of input rows and
# the right edge of output rows, reproducing the imnodes left-in/right-out wiring.
function draw_pin(client, id, kind)
    pivot = kind == ne.PinKind_Input ? ImVec2(0f0, 0.5f0) : ImVec2(1f0, 0.5f0)
    ne.PushStyleVar(ne.StyleVar_PivotAlignment, pivot)
    ne.BeginPin(pin_handle(client, id), kind)
    draw_pin_icon()
    ne.EndPin()
    ne.PopStyleVar()
end

# Show a tooltip at the mouse. An ImGui tooltip created while drawing a node is
# captured by the editor's zoomed canvas and lands in the wrong place; Suspend()ing
# to escape it corrupts the node/pin draw list. So inside the editor we draw the
# tooltip directly on the foreground draw list, which is screen space and never
# touched by the canvas. Outside the editor, a normal tooltip is fine.
function node_tooltip(text)
    if ne.GetCurrentEditor() == C_NULL
        ig.SetTooltip(text)
        return
    end

    # While the canvas is active GetMousePos() is in zoomed canvas space; map it
    # back to screen space for the (unscaled) foreground draw list. Scale the text
    # and box by the canvas magnification so they match the zoomed node text, but
    # clamp to between the default and twice the default so it stays readable.
    scale = clamp(ne.current_scale(), 1f0, 2f0)
    font = ig.GetFont()
    font_size = ig.GetFontSize() * scale

    draw_list = ig.GetForegroundDrawList()
    mouse = ne.CanvasToScreen(ig.GetMousePos())
    text_size = ig.CalcTextSize(text)
    pad = ImVec2(8 * scale, 6 * scale)
    pos = ImVec2(mouse.x + 16 * scale, mouse.y + 8 * scale)

    # The canvas leaves the foreground list clipped to the (zoom-shrunk) local
    # viewport, which culls the tooltip when zoomed in; clip to the full screen.
    ig.PushClipRectFullScreen(draw_list)
    ig.AddRectFilled(draw_list, (pos.x - pad.x, pos.y - pad.y),
                     (pos.x + text_size.x * scale + pad.x, pos.y + text_size.y * scale + pad.y),
                     IM_COL32(35, 35, 40, 240), 4f0 * scale)
    # The foreground list is unscaled screen space, but when zoomed in the frame
    # density is cranked up for the canvas-magnified node text; that bakes this
    # tooltip oversized and minifies it (aliasing). Bake at the displayed size.
    frame_density = ig.GetFontRasterizerDensity()
    ig.SetFontRasterizerDensity(1f0)
    ig.AddText(draw_list, font, font_size, pos, IM_COL32(230, 230, 230, 255), text)
    ig.SetFontRasterizerDensity(frame_density)
    ig.PopClipRect(draw_list)
end

# Node names in front-to-back order so the top node under the mouse wins input on
# overlapping widgets: ImGui gives the first-submitted overlapping item the click,
# and the editor keeps the active/selected node last in its order. New nodes the
# editor hasn't seen yet are appended.
function node_draw_order(client, ctx_state)
    n = Int(ne.GetNodeCount())
    if n == 0
        return collect(keys(ctx_state))
    end

    buf = Vector{UInt}(undef, n)
    GC.@preserve buf ne.GetOrderedNodeIds(Ptr{ne.NodeId}(pointer(buf)), n)

    id_to_name = Dict{UInt, String}(UInt(var_data["id"]) => name for (name, var_data) in ctx_state)
    ordered = String[]
    for id in Iterators.reverse(buf)
        name = get(id_to_name, id, nothing)
        if !isnothing(name)
            push!(ordered, name)
        end
    end
    for name in keys(ctx_state)
        if !(name in ordered)
            push!(ordered, name)
        end
    end
    return ordered
end

## Helper functions for the GUI

function draw_revise()
    can_revise = length(Revise.revision_queue) > 0
    @Disabled !can_revise begin
        if ig.Button(can_revise ? "Revise*" : "Revise")
            Revise.retry()

            client = state[].client
            if client.status == RemoteStatus_Connected
                revise_engine(state[])
            end
        end
    end
end

function draw_main_menubar()
    client = state[].client

    if ig.BeginMenuBar()
        draw_revise()

        can_sync = client.status == RemoteStatus_Connected && !client.syncing
        @Disabled !can_sync begin
            if ig.Button("Sync")
                @guiasync sync_files()
            end
            if client.syncing
                Spinner()
            end
        end

        if ig.BeginMenu("Tools")
            if ig.BeginMenu("Demos")
                @c MenuItem("ImGui demo", &state[].show_imgui_demo)
                ig.EndMenu()
            end

            @c MenuItem("ImGui metrics", &state[].show_imgui_metrics)
            @c MenuItem("Stack tool", &state[].show_stacktool)
            @c MenuItem("Debug log", &state[].show_debug_log)
            @c MenuItem("State inspector", &state[].show_state_inspector)
            if @c MenuItem("Engine logs", &state[].show_engine_logs)
                state[].select_engine_logs = true
            end

            ig.EndMenu()
        end

        ig.EndMenuBar()
    end
end

function draw_parameter_widget(name, param::Parameter{Float64})
    ret = @c ig.InputDouble("##$(name)", &param.value, 0.0, 0.0, "%.3f0", ig.ImGuiInputTextFlags_EnterReturnsTrue)

    return ret, param.value
end

function draw_parameter_widget(name, param::Parameter{Int})
    int32_ref = Ref(Int32(param.value))
    ret = ig.InputInt("##$(name)", int32_ref, 1, 100, ig.ImGuiInputTextFlags_EnterReturnsTrue)
    if ret
        param.value = Int(int32_ref[])
    end

    return ret, param.value
end

function draw_parameter_widget(name, param::Parameter{String})
    return SafeInputText("##$(name)"; current_text=param.value)
end

# True if `param_name` is a fully-qualified "<group>.<field>" name belonging to
# a group node in the current context.
function is_group_param(client, param_name::String)
    dot = findfirst('.', param_name)
    if isnothing(dot)
        return false
    end
    group_name = param_name[1:dot-1]
    var_data = get(client.context.context_state, group_name, nothing)
    return !isnothing(var_data) && get(var_data, "type", nothing) === :group
end

function draw_parameter_widget(name, param::Parameter{Vector{String}})
    ig.Text("Vector{String}")

    return false, nothing
end

function draw_numeric_vector_widget(name, param::Parameter{Vector{T}}) where {T <: Number}
    current = join(param.value, ", ")
    edited, new_text = SafeInputText("##$(name)"; current_text=current)
    if !edited
        return false, nothing
    end
    parts = [strip(s) for s in split(new_text, ","; keepempty=false)]
    parsed = [tryparse(T, p) for p in parts]
    if any(isnothing, parsed)
        return false, nothing
    end
    return true, T[parsed...]
end

function draw_parameter_widget(name, param::Parameter{Vector{Int}})
    return draw_numeric_vector_widget(name, param)
end

function draw_parameter_widget(name, param::Parameter{Vector{Float64}})
    return draw_numeric_vector_widget(name, param)
end

# (lo, hi) range editor. We only commit a new value once lo < hi so the engine
# isn't bombarded with half-edited input. Tracking of "user-pinned vs auto" is
# handled engine-side via the parameter's `set_by_user` flag.
function draw_parameter_widget(name, param::Parameter{Tuple{Float64, Float64}})
    lo, hi = param.value
    buf = Cdouble[lo, hi]
    edited = ig.InputScalarN("##$(name)", ig.ImGuiDataType_Double, buf, 2,
                             C_NULL, C_NULL, "%.3f0", ig.ImGuiInputTextFlags_EnterReturnsTrue)
    if !edited || !(buf[1] < buf[2])
        return false, nothing
    end
    return true, (buf[1], buf[2])
end

function draw_parameter_widget(name, param::Parameter{OptionalDims})
    ps = state[].client.parameter_states[param.name]::OptionalDimsState

    checkbox_changed = @c ig.Checkbox("All##$(name)", &ps.all_dims)
    if checkbox_changed
        if ps.all_dims
            return true, OptionalDims()
        end
    end

    if !ps.all_dims
        current = join(param.value.dims, ", ")
        text = isempty(ps.pending_text) ? current : ps.pending_text
        if ps.pending_text == current
            ps.pending_text = ""
        end
        ig.SetNextItemWidth(ig.GetContentRegionAvail().x)
        edited, new_text = SafeInputText("##dims_$(name)"; current_text=text)

        if edited || checkbox_changed
            ps.pending_text = new_text
            parts = [strip(s) for s in split(new_text, ","; keepempty=false)]
            dims = try
                Int[parse(Int, p) for p in parts]
            catch
                String[parts...]
            end
            return true, OptionalDims(dims)
        end
    end

    return false, nothing
end

function draw_parameter_widget(name, param::Parameter{KaraboDevice})
    client = state[].client
    dep_key = hash(param.name)
    dep_state = get!(client.karabo_dep_states, dep_key, KaraboDepTextState())
    device_props = if isnothing(dep_state.device)
        DeviceProperties()
    else
        get_source_properties(client, dep_state.device)
    end

    device = param.value
    text = "$(device.topic)//$(device.name)"
    edited, new_text = KaraboDepText("param-$(param.name)", text, dep_state,
                                     client.source_list, device_props, client; device_only=true)
    if edited
        new_device = KaraboDevice(new_text)
        if isempty(new_device.topic)
            idx = findfirst(s -> s.name == new_device.name, client.source_list)
            if !isnothing(idx)
                new_device = KaraboDevice(client.source_list[idx].topic, new_device.name)
            end
        end
        return true, new_device
    end

    return false, nothing
end

# Draw a dependency editor (type selector + autocomplete text field).
# Returns (edited::Bool, new_dep::Dependency). Used for both dependency pins
# and Parameter{Dependency} widgets.
function draw_dep_editor(label, dep::Dependency, dep_id::Integer;
                         device_only::Bool=false, variable_name::String="")
    client = state[].client
    dep_state = get!(client.dep_text_states, dep_id) do
        DepTextState(; is_karabo=dep.kind == DepKind_Karabo)
    end
    device_props = if isnothing(dep_state.karabo_state.device)
        DeviceProperties()
    else
        get_source_properties(client, dep_state.karabo_state.device)
    end
    return DepText(label, dep, dep_state, client.source_list, device_props,
                   client.variable_names, client; device_only, variable_name)
end

function draw_parameter_widget(name, param::Parameter{Dependency})
    dep = param.value
    dep_id = hash(param.name)
    edited, new_dep = draw_dep_editor("param-dep-$(param.name)", dep, dep_id)
    if edited
        return true, new_dep
    end
    return false, nothing
end

function draw_parameter_widget(name, param::Parameter{RectROI})
    roi = param.value
    text = "($(roi.corner_x), $(roi.corner_y), $(roi.width), $(roi.height))"
    buf = Vector{UInt8}(undef, length(text) + 1)
    Util.strcpy!(buf, text)
    ig.InputText("##$(name)", buf, length(buf), ig.ImGuiInputTextFlags_ReadOnly)
    return false, nothing
end

function draw_parameter_widget(name, param::Parameter{Bool})
    changed = @c ig.Checkbox("##$(name)", &param.value)
    return changed, param.value
end

function get_variable_typeinfo(name)
    variable_data = state[].client.variable_data
    if haskey(variable_data, name)
        data = variable_data[name].data
        if data isa ArrayMetadata
            return "$(data.eltype)$(Tuple(data.size))"
        else
            T = eltype(data)
            return "$T$(size(data))"
        end
    else
        return ""
    end
end

# Number of dimensions of the array (or array metadata) backing a variable, or
# nothing for non-array data (scalars).
function variable_ndims(name)
    variable_data = state[].client.variable_data
    if !haskey(variable_data, name)
        return nothing
    end
    data = variable_data[name].data
    if data isa ArrayMetadata
        return length(data.size)
    elseif data isa CircularBuffer
        return nothing
    else
        return ndims(data)
    end
end

# Plot button that's disabled with a tooltip when the variable has more than
# two dimensions. Returns true on click. `button` is the imgui button function
# to use (e.g. ig.Button or ig.SmallButton).
function plot_button(label, name; button=ig.Button)
    nd = variable_ndims(name)
    too_many_dims = !isnothing(nd) && nd > 2

    if too_many_dims
        ig.BeginDisabled()
    end

    clicked = button(label)

    if too_many_dims
        ig.EndDisabled()
        if ig.IsItemHovered(ig.ImGuiHoveredFlags_AllowWhenDisabled)
            ig.SetTooltip("Plotting arrays with more than 2 dimensions is not supported")
        end
    end

    return clicked
end

function clear_variable_data(store)
    if store.data isa AbstractVector
        empty!(store.data)
    end
    if !isnothing(store.scalar_tids)
        empty!(store.scalar_tids)
    end
end

function clear_variables()
    client = state[].client

    for store in values(client.variable_data)
        clear_variable_data(store)
    end

    for plot in client.plots
        clear_plot(plot)
    end
end

function draw_device_tree(device_tree)
    if isempty(device_tree)
        ig.TextDisabled("No devices loaded")
        return
    end

    n_devices = sum(length(devs) for (_, devs) in device_tree)
    n_topics = length(device_tree)
    if ig.TreeNode("Devices ($n_devices across $n_topics topics)##device-tree")
        for (topic, devices) in device_tree
            if ig.TreeNode("$topic ($(length(devices)))##topic-$topic")
                for (name, info_pairs) in devices
                    class_id_pair = findfirst(p -> p.first == "classId", info_pairs)
                    class_id = isnothing(class_id_pair) ? "" : info_pairs[class_id_pair].second
                    if ig.TreeNode("$name##dev-$name")
                        for (key, value) in info_pairs
                            ig.Text("$key: $value")
                        end
                        ig.TreePop()
                    else
                        ig.SameLine()
                        ig.TextDisabled(class_id)
                    end
                end
                ig.TreePop()
            end
        end
        ig.TreePop()
    end
end

function get_source_properties(client, device_name)
    idx = findfirst(s -> s.name == device_name, client.source_list)
    isnothing(idx) && return DeviceProperties()

    topic = client.source_list[idx].topic
    key = (topic, device_name)
    return get!(client.source_properties, key) do
        id = send(client, GetDeviceSchema(topic, device_name))
        client.device_schema_requests[key] = id
        DeviceProperties()
    end
end

# Draw a single parameter with appropriate width, and send a change message
# if modified.
function draw_parameter(name, param; min_node_width=150)
    ig.Text(name * ":")
    ig.SameLine()
    ig.SetNextItemWidth(round(Int, min_node_width * 1.5))
    modified, new_value = draw_parameter_widget(name, param)
    if modified
        state[].client.pending_source_edit = param.name
        change_parameter(Parameter(param.name, new_value))
    end
    return modified, new_value
end

# Draw the parameters section of a variable node. Can be called from custom
# draw_variable_content() methods to include the default parameter UI.
function draw_parameters(var_data)
    if haskey(var_data, "parameters")
        ig.Text("Parameters:")
        for (param_name, param) in var_data["parameters"]
            draw_parameter(param_name, param)
        end
    end
end

# Specialize on Val{Symbol("ModulePath.function_name")} to draw custom content
# inside a variable node. Called after the titlebar and before parameters.
# Return a gui state object to persist custom state across frames, or nothing.
draw_variable_content(::Val, name, var_data, gui_state) = nothing

# Specialize on Val{Symbol("ModulePath.PostprocessorType")} to draw a custom
# parameter UI for a postprocessor. Default: list every parameter.
function draw_postprocessor_params(::Val, pp, min_node_width)
    for (param_name, param) in pp.params
        draw_parameter(param_name, param; min_node_width)
    end
end

# Draws a variable node. The node shell (titlebar, dependencies, outputs) is
# always the same, but draw_variable_content() is called inside to allow
# custom rendering for specific variables.
function draw_variable(name, var_data)
    client = state[].client
    min_node_width = 150
    variable_store = get(client.variable_data, name, nothing)

    ig.PushID(name)
    handle = node_handle(client, var_data["id"])
    ne.BeginNode(handle)

    # Right-align the output/postprocessor pins to the widest left-anchored row.
    # `content_measured` accumulates each row's width (via origin-independent
    # GetItemRectSize) this frame; `content_width` reuses last frame's value to
    # place the pins. The pins are never measured, so they can't ratchet the node
    # ever-wider.
    content_measured = Float32(min_node_width)
    content_width = get(client.ne_node_content_widths, var_data["id"], Float32(min_node_width))

    disable_node = client.context.pipeline_status ∉ (PipelineStatus_Stopped, PipelineStatus_Started) ||
                   !isnothing(client.pending_parameter_change)
    @Disabled disable_node begin
        # Draw the titlebar at a fixed 1.5x canvas size (1.5*current_scale() keeps
        # the local size constant across zoom) so it doesn't grow the node's
        # canvas-space width when zoomed out and strand the right-aligned pins.
        edited, new_name = ne.@with_font_scale 1.5f0 * ne.current_scale() ElidedText("var-name-$(name)", name;
            editable=true, validator=variable_name_validator(name))
        if edited
            @guiasync rename_variable(state[], name, new_name)
        end
        # Bottom of the title, used to size the header background drawn after
        # EndNode. The Dummy adds breathing room between the header and content.
        title_bottom = ig.GetItemRectMax().y
        content_measured = max(content_measured, ig.GetItemRectSize().x)
        ig.Dummy(0, 6)
        # Group the custom content so we can measure its full width; the helpers
        # draw several rows and GetItemRectMax alone would only see the last one.
        ig.BeginGroup()
        origin = var_data["origin"]
        gui_state = get(client.variable_gui_states, name, nothing)
        new_gui_state = draw_variable_content(Val(Symbol(origin)), name, var_data, gui_state)
        if !isnothing(new_gui_state) && !haskey(client.variable_gui_states, name)
            client.variable_gui_states[name] = new_gui_state
        end

        if var_data["draw_parameters"]
            draw_parameters(var_data)
        end
        ig.EndGroup()
        content_measured = max(content_measured, ig.GetItemRectSize().x)

        ig.Dummy(min_node_width, 20)

        # Draw dependencies
        deps = var_data["dependencies"]
        for (dep_id, dep_pair) in deps
            arg_name, dep = dep_pair
            # Don't draw pins for parameters
            if dep isa Parameter
                continue
            end

            dep_ts = get!(client.dep_text_states, dep_id) do
                DepTextState(; is_karabo=dep isa Dependency && dep.kind == DepKind_Karabo)
            end

            # The pin wraps only the icon so its hover rect doesn't span the whole
            # row; the label/editor are drawn after EndPin on the same line.
            draw_pin(client, dep_id, ne.PinKind_Input)
            ig.SameLine()
            # Group the post-pin content so we can size the row from it. The pin
            # icon to its left is a fixed advance (icon + SameLine spacing).
            ig.BeginGroup()
            if var_data["type"] == :group
                label = get(var_data["dep_field_names"], dep_id, arg_name)
                ig.Text(label * ":")
                ig.SameLine()
            end
            edited, new_dep = draw_dep_editor("dep-$(dep_id)", dep, dep_id; variable_name=name)
            ig.EndGroup()
            pin_advance = ig.GetTextLineHeight() + unsafe_load(ig.GetStyle().ItemSpacing.x)
            content_measured = max(content_measured, pin_advance + ig.GetItemRectSize().x)
            if edited
                # For group nodes the kwarg in the constructor uses the group
                # struct field name, not the @Variable's arg name.
                target_arg = arg_name
                if var_data["type"] == :group
                    target_arg = get(var_data["dep_field_names"], dep_id, arg_name)
                end
                @guiasync rename_dep(state[], name, target_arg, dep, new_dep)
            end
        end
    end # @Disabled

    ig.Dummy(min_node_width, 10)

    ig.TextDisabled("Outputs")
    draw_list = ig.GetWindowDrawList()
    start_pos = ig.GetCursorScreenPos()
    gray = ig.IM_COL32(100, 100, 100, 255)
    ig.AddLine(draw_list, start_pos, (start_pos.x + min_node_width / 2f0, start_pos.y), gray, 2)
    ig.Dummy(min_node_width, 2)

    # Draw outputs
    for output in var_data["outputs"]
        label = output.label
        output_name = isempty(label) ? name : "$(name).$(label)"
        pin_start = ig.GetCursorPos()

        # Measure the label group before placing the pin. BeginPin leaves the
        # cursor's max-x at the far-right pin position, so measuring after it would
        # make content_width self-sustaining (the node would never shrink). The pin
        # is drawn afterwards via SetCursorPos.
        ig.BeginGroup()
        typestr = get_variable_typeinfo(output_name)
        if !isempty(typestr)
            label = isempty(label) ? typestr : "$(label) - $(typestr)"
        end

        if !isempty(label)
            if haskey(client.variable_data, output_name)
                if plot_button("$(label)###$(output_name)-plot_button", output_name)
                    push!(client.plots, variable_plot(output_name, client.plot_counter))
                    client.plot_counter += 1
                end
            else
                ig.Text(label)
            end
        end

        if !output.is_subvariable
            rate = if haskey(client.variable_data, output_name)
                client.variable_data[output_name].update_rate
            else
                get(client.context.input_rates, output_name, nothing)
            end
            if !isnothing(rate)
                ig.SameLine()
                ig.TextDisabled(@sprintf "%.2f Hz" rate)
            end
        end

        # Advertised plot specs are openable in addition to the default plot.
        if haskey(client.variable_data, output_name)
            specs = client.variable_data[output_name].plot_specs
            if !isempty(specs)
                ig.Indent()
                for spec in specs
                    if ig.SmallButton("\uf201 $(spec.name)###$(output_name)-spec-$(spec.name)")
                        push!(client.plots, spec_plot(output_name, spec.name, client.plot_counter))
                        client.plot_counter += 1
                    end
                end
                ig.Unindent()
            end
        end
        ig.EndGroup()
        content_measured = max(content_measured, ig.GetItemRectSize().x)
        after_label = ig.GetCursorPos()

        # Right-aligned output pin on the label's row. Drawing it (icon Dummy) after
        # the rightward SetCursorPos validates the extent (no boundary assert).
        icon_w = ig.GetTextLineHeight()
        ig.SetCursorPos((pin_start.x + content_width - icon_w, pin_start.y))
        draw_pin(client, output.id, ne.PinKind_Output)
        # Continue below both the label and the pin icon so rows don't overlap.
        # The trailing Dummy validates the SetCursorPos extent so ImGui's boundary
        # check doesn't warn.
        ig.SetCursorPos((pin_start.x, max(after_label.y, pin_start.y + icon_w)))
        ig.Dummy(0f0, 0f0)
    end

    # Draw postprocessors
    postprocessors = get(var_data, "postprocessors", [])
    if !isempty(postprocessors)
        ig.Dummy(min_node_width, 8)
        ig.TextDisabled("Postprocessors")
        pp_sep_pos = ig.GetCursorScreenPos()
        ig.AddLine(draw_list, pp_sep_pos, (pp_sep_pos.x + min_node_width / 2f0, pp_sep_pos.y), gray, 2)
        ig.Dummy(min_node_width, 2)

        for pp in postprocessors
            label = pp.display_name * pp.tree_id_suffix
            pin_start = ig.GetCursorPos()
            icon_w = ig.GetTextLineHeight()

            # The pin wraps only the icon (right-aligned to the node edge) so its
            # hover rect doesn't span the whole row; the tree is drawn after EndPin
            # from the left.
            ig.SetCursorPos((pin_start.x + content_width - icon_w, pin_start.y))
            draw_pin(client, pp.id, ne.PinKind_Output)
            ig.SetCursorPos(pin_start)

            # Constrain the tree widget width with the old Columns API, otherwise
            # the TreeNode's clickable area stretches to the window edge (see the
            # imgui-node-editor widgets example).
            ig.BeginColumns("##pp-$(pp.id)", 2,
                            ig.ImGuiOldColumnFlags_NoBorder | ig.ImGuiOldColumnFlags_NoResize |
                            ig.ImGuiOldColumnFlags_NoPreserveWidths | ig.ImGuiOldColumnFlags_NoForceWithinWindow)
            ig.SetColumnWidth(0, content_width - icon_w)

            expanded = ig.TreeNode(label)
            if haskey(client.variable_data, pp.name)
                typestr = get_variable_typeinfo(pp.name)
                if !isempty(typestr)
                    ig.SameLine()
                    if plot_button("$(typestr)$(pp.tree_id_suffix)_plot", pp.name; button=ig.SmallButton)
                        push!(client.plots, variable_plot(pp.name, client.plot_counter))
                        client.plot_counter += 1
                    end
                end
            end

            if expanded
                if isempty(pp.params)
                    ig.TextDisabled("(no parameters)")
                else
                    draw_postprocessor_params(Val(Symbol(pp.origin)), pp, min_node_width)
                end
                ig.TreePop()
            end

            ig.EndColumns()
        end
    end

    # Stash the widest content row for next frame's pin alignment. Postprocessors
    # are excluded: their tree column is sized to content_width, so measuring it
    # would feed the imposed width back and re-introduce the ratchet.
    client.ne_node_content_widths[var_data["id"]] = content_measured

    ne.EndNode()

    # Dark blue header background behind the title (following builders.cpp
    # BlueprintNodeBuilder::End): drawn after EndNode on the node's background draw
    # list so it sits above the node fill but below the title text.
    node_pos = ne.GetNodePosition(handle)
    node_size = ne.GetNodeSize(handle)
    border = unsafe_load(ne.GetStyle().NodeBorderWidth)
    # Shrink the radius with the inset so the band's corner stays concentric with
    # the border's inner edge.
    rounding = max(0f0, unsafe_load(ne.GetStyle().NodeRounding) - border)
    # Match the band's alpha to the node body so it's translucent like the rest.
    node_alpha = round(Int, unsafe_load(ne.GetStyle().Colors)[ne.StyleColor_NodeBg + 1].w * 255)
    # Inset by the border width so the band stays inside the node's (selection)
    # border instead of painting over it.
    ig.AddRectFilled(ne.GetNodeBackgroundDrawList(handle),
                     (node_pos.x + border, node_pos.y + border),
                     (node_pos.x + node_size.x - border, title_bottom),
                     IM_COL32(55, 62, 82, node_alpha), rounding, ig.ImDrawFlags_RoundCornersTop)

    ig.PopID()
end

# Link color for a channel-fill ratio (load ∈ [0, 1]). Ramps from muted green
# (empty) through orange (half-full) to bright red (at capacity). The green
# ceiling is lowered so idle channels don't glare.
function link_load_color(load)
    green_ceiling = 0xa0 / 0xff
    r = load < 0.5 ? 2 * load : 1
    g = load < 0.5 ? green_ceiling : green_ceiling * 2 * (1 - load)
    return ImVec4(r, g, 0, 1)
end

function draw_routing_rules()
    client = state[].client

    topics = sort!(collect(keys(client.trainmatchers)))

    routing_rules_pending = is_pending(client, client.routing_rules_set_request)
    @Disabled routing_rules_pending begin
        if ig.Button("Add rule")
            default_topic = isempty(topics) ? "" : first(topics)
            push!(client.routing_rules, RoutingRule(default_topic, "*", ""))
            set_routing_rules(client, client.routing_rules)
        end
        ig.SameLine()
        if ig.Button("Refresh trainmatchers")
            get_trainmatchers(client)
        end
        if client.trainmatchers_request_status == RequestStatus_Waiting
            ig.SameLine()
            Spinner()
        end
        if routing_rules_pending
            ig.SameLine()
            Spinner()
        end

        if client.routing_rules_request_status == RequestStatus_Waiting
            Spinner("Waiting for routing rules")
        else
            ig.BeginTable("##routing-rules", 4,
                          ig.ImGuiTableFlags_Borders | ig.ImGuiTableFlags_RowBg |
                          ig.ImGuiTableFlags_Resizable)
            ig.TableSetupColumn("", ig.ImGuiTableColumnFlags_WidthFixed |
                                     ig.ImGuiTableColumnFlags_NoResize)
            ig.TableSetupColumn("Topic")
            ig.TableSetupColumn("Source")
            ig.TableSetupColumn("Input (trainmatcher device)")
            ig.TableHeadersRow()

            # Mutations are deferred to after the iteration loop so we don't
            # shift indices out from under ourselves mid-row.
            rules_changed = false
            delete_idx = nothing
            move_idx = nothing
            for (i, rule) in enumerate(client.routing_rules)
                ig.TableNextRow()

                # Column 1: row controls (drag-to-reorder grip + delete button).
                ig.TableNextColumn()
                grip_icon = "\uf58d"
                ig.PushStyleVar(ig.ImGuiStyleVar_SelectableTextAlign, ImVec2(0.5, 0.5))
                ig.Selectable("$(grip_icon)##rule-grip-$i", false, 0,
                              ImVec2(ig.CalcTextSize(grip_icon).x + 4, ig.GetFrameHeight()))
                ig.PopStyleVar()
                if ig.IsItemHovered()
                    ig.SetTooltip("Drag to reorder")
                end

                if ig.BeginDragDropSource()
                    payload = Ref{Cint}(i)
                    ig.SetDragDropPayload("ROUTING_RULE_ROW", payload, sizeof(Cint))
                    ig.Text(rule.topic * "  /  " * rule.source)
                    ig.EndDragDropSource()
                end
                if ig.BeginDragDropTarget()
                    accepted = ig.AcceptDragDropPayload("ROUTING_RULE_ROW")
                    if accepted != C_NULL
                        from = unsafe_load(Ptr{Cint}(unsafe_load(accepted).Data))
                        if from != i
                            move_idx = (Int(from), i)
                        end
                    end
                    ig.EndDragDropTarget()
                end

                ig.SameLine()
                if ig.Button("\uf2ed##rule-$i")
                    delete_idx = i
                end
                if ig.IsItemHovered()
                    ig.SetTooltip("Delete rule")
                end

                # Column 2: topic glob (free-form text).
                ig.TableNextColumn()
                ig.SetNextItemWidth(-1)
                edited, new_topic = SafeInputText("##rule-topic-$i"; current_text=rule.topic)
                if edited
                    client.routing_rules[i] = RoutingRule(new_topic, rule.source, rule.input)
                    rules_changed = true
                end

                # Column 3: source autocomplete, restricted to devices in the row's topic.
                ig.TableNextColumn()
                ig.SetNextItemWidth(-1)
                src_state = get!(client.routing_rule_source_states, i, KaraboDepTextState())
                src_props = if isnothing(src_state.device)
                    DeviceProperties()
                else
                    get_source_properties(client, src_state.device)
                end
                filtered_sources = get(client.sources_by_topic, rule.topic, SourceInfo[])
                edited, new_source = KaraboDepText("##rule-source-$i", rule.source, src_state,
                                                   filtered_sources, src_props, client;
                                                   allow_slow=false)
                if edited
                    client.routing_rules[i] = RoutingRule(rule.topic, new_source, rule.input)
                    rules_changed = true
                end

                # Column 4: trainmatcher combo, populated with devices from the row's topic.
                # A yellow warning icon appears when the selected device isn't whitelisted
                # in the topic's webproxy.
                ig.TableNextColumn()
                tm_names = sort!(get(client.trainmatchers, rule.topic, String[]))
                tm_warn = rule.input in tm_names &&
                          !(KaraboDevice(rule.topic, rule.input) in client.whitelisted_trainmatchers)

                # Reserve space at the right edge of the cell for the warning icon
                # so the combo doesn't push it out of the column.
                warning_icon = "\uf06a"
                icon_w = ig.CalcTextSize(warning_icon).x
                ig.SetNextItemWidth(-(icon_w + 8))
                tm_idx = findfirst(==(rule.input), tm_names)
                tm_sel = Ref(Cint(isnothing(tm_idx) ? -1 : tm_idx - 1))
                if CopyableCombo("rule-input-$i", tm_names, tm_sel)
                    new_input = tm_names[tm_sel[] + 1]
                    client.routing_rules[i] = RoutingRule(rule.topic, rule.source, new_input)
                    rules_changed = true
                end

                if tm_warn
                    ig.SameLine()
                    ig.TextColored(ImVec4(1.0, 0.7, 0.0, 1.0), warning_icon)
                    if ig.IsItemHovered() && ig.BeginTooltip()
                        ig.Text("""This trainmatcher is not in the webproxy whitelist. That means
                                   that it cannot be used with XFA because it cannot be automatically
                                   reconfigured.

                                   To use $(rule.input), you have to add it to the
                                   `devices` property of `karabo/WebProxy/device` in the $(rule.topic) topic.""")
                        ig.EndTooltip()
                    end
                end
            end

            # Apply deferred reorder/delete now that the row loop is done.
            if !isnothing(move_idx)
                from, to = move_idx
                rule = client.routing_rules[from]
                deleteat!(client.routing_rules, from)
                insert!(client.routing_rules, to, rule)
                rules_changed = true
            end

            if !isnothing(delete_idx)
                deleteat!(client.routing_rules, delete_idx)
                rules_changed = true
            end

            if rules_changed
                set_routing_rules(client, client.routing_rules)
            end

            ig.EndTable()
        end
    end
end

function draw_dag()
    client = state[].client
    context = client.context
    ctx_state = context.context_state

    ig.Dummy(0, 10)
    @Disabled isempty(ctx_state) || context.pipeline_status != PipelineStatus_Stopped begin
        if ig.Button(" Start ")
            start(state[])
        end
    end

    ig.SameLine()

    @Disabled context.pipeline_status != PipelineStatus_Started begin
        if ig.Button(" Stop ")
            stop(state[])
        end
    end

    ig.SameLine()
    @Disabled isempty(client.variable_data) begin
        if ig.Button("Clear all")
            clear_variables()
        end
    end

    ig.SameLine()
    changing_states = (PipelineStatus_Starting, PipelineStatus_Stopping, PipelineStatus_LoadingContext)
    @Disabled context.pipeline_status in changing_states begin
        if ig.Button("Load context")
            load_context(state[])
            restore_plots(state[])
        end
    end

    ig.SameLine()
    @Disabled isempty(client.variable_data) begin
        if ig.Button("Correlate")
            push!(client.plots, correlation_plot(client.plot_counter))
            client.plot_counter += 1
        end
    end

    if context.pipeline_status in changing_states
        ig.SameLine()
        Spinner()
    end

    ig.SameLine()
    ig.SetCursorPosX(ig.GetCursorPos().x + ig.GetContentRegionAvail().x - 100)
    if ig.Button("Add variable")
        ig.OpenPopup("add_variable_popup")
    end
    if ig.BeginPopup("add_variable_popup")
        if ig.Selectable("Karabo source")
        end
        ig.EndPopup()
    end

    ig.Dummy(0, 10)

    # (Re)create the editor whenever the loaded context changes so it restores
    # that context's saved layout. Persist the outgoing context's layout first.
    if client.ne_editor_path != client.context_path
        if client.ne_editor != C_NULL
            save_node_editor_state(client.ne_editor_path, client.ne_settings)
            ne.DestroyEditor(client.ne_editor)
        end
        client.ne_settings = load_node_editor_state(client.context_path)
        create_node_editor!(client)
    end

    # Left edge of the canvas in screen space, used to keep deferred popups from
    # spilling off the side of the window.
    canvas_min_x = ig.GetCursorScreenPos().x

    ne.SetCurrentEditor(client.ne_editor)
    ne.Begin("dag-editor")

    # The editor zooms by scaling the draw list, which blurs text baked at the
    # base size. Match the font rasterizer density to the magnification (snapped
    # up to a power of two so the bake cache stays warm) to keep text crisp.
    magnification = ne.current_scale()
    ig.SetFontRasterizerDensity(exp2(ceil(log2(max(magnification, 1f0)))))

    for name in node_draw_order(client, ctx_state)
        var_data = ctx_state[name]
        ig.PushID(name)

        draw_variable(name, var_data)

        # Apply the auto-layout position only to nodes the editor didn't restore
        # (new nodes sit at the origin); restored nodes keep their saved spot.
        pos = context.node_positions[name]
        if pos != Point2d(-1, -1)
            handle = node_handle(client, var_data["id"])
            editor_pos = ne.GetNodePosition(handle)
            if editor_pos.x == 0 && editor_pos.y == 0
                ne.SetNodePosition(handle, (pos.x, pos.y))
            end
            context.node_positions[name] = Point2d(-1, -1)
        end

        ig.PopID()
    end

    # The dep-kind and CopyableCombo dropdowns are drawn here, after all nodes,
    # under Suspend/Resume so they escape the node canvas and position in screen
    # space (a popup opened mid-node lands in the wrong place). Only one of each is
    # open at a time; the in-node widgets set the request and a one-shot trigger.
    ne.Suspend()
    if !isnothing(client.dep_kind_popup)
        popup_label, dep_state = client.dep_kind_popup
        if client.dep_kind_popup_trigger
            ig.OpenPopup(popup_label)
            client.dep_kind_popup_trigger = false
        end
        if ig.BeginPopup(popup_label)
            for (is_karabo, kind_label) in ((true, "Karabo"), (false, "Variable"))
                if ig.Selectable(kind_label, dep_state.is_karabo == is_karabo)
                    if dep_state.is_karabo != is_karabo
                        dep_state.is_karabo = is_karabo
                        dep_state.wants_focus = true
                        # Reset karabo state when switching
                        dep_state.karabo_state.cursor_pos = -1
                        dep_state.karabo_state.device = nothing
                        dep_state.karabo_state.wanted_text = nothing
                    end
                    ig.CloseCurrentPopup()
                end
            end
            ig.EndPopup()
        else
            client.dep_kind_popup = nothing
        end
    end

    popup = client.combo_popup
    if !isnothing(popup.label)
        popup_id = "combo-popup-$(popup.label)"
        if popup.trigger
            ig.OpenPopup(popup_id)
            popup.trigger = false
        end
        anchor = ImVec2(max(popup.anchor.x, canvas_min_x), popup.anchor.y)
        ig.SetNextWindowPos(anchor)
        new_idx = draw_combo_popup(popup_id, popup.items, popup.width)
        if !isnothing(new_idx)
            popup.result_label = popup.label
            popup.result_index = new_idx
            popup.label = nothing
        elseif !ig.IsPopupOpen(popup_id)
            popup.label = nothing
        end
    end
    ne.Resume()

    channel_stats = context.channel_stats
    for var_data in values(ctx_state)
        for link in var_data["links"]
            stat = get(channel_stats, link.channel_key, nothing)
            colored = !isnothing(stat) && stat.capacity > 0
            color = colored ? link_load_color(stat.size / stat.capacity) : ImVec4(1, 1, 1, 1)
            ne.Link(link_handle(client, link.id), pin_handle(client, link.start_id),
                    pin_handle(client, link.end_id), color)
        end
    end

    ne.End()
    ig.SetFontRasterizerDensity(1f0)
    ne.SetCurrentEditor(C_NULL)

    # Timer to save the current settings periodically. Mostly useful for the
    # node positions.
    framerate = round(Int, unsafe_load(ig.GetIO().Framerate))
    if framerate > 0 && ig.GetFrameCount() % (5 * framerate) == 0
        if !isempty(ctx_state)
            save_settings(client)
        end
    end
end

function draw_ssh_auth()
    client = state[].client

    for (hop_idx, ssh_state) in enumerate(client.ssh_hops)
        if ssh_state.auth_state == ssh.AuthStatus_Success
            ig.BulletText("Successfully authenticated to $(ssh_state.address)")
            continue
        elseif isnothing(ssh_state.session)
            if ssh_state.auth_state == :connecting
                ig.BulletText("Connecting to $(ssh_state.address) ")
                ig.SameLine()
                Spinner()
            else
                ig.BulletText("Next hop: $(ssh_state.address)")
            end
            continue
        end

        host = if isnothing(ssh_state.session) || !isopen(ssh_state.session)
            ssh_state.address
        else
            "$(ssh_state.session.user)@$(ssh_state.address)"
        end

        ig.BulletText("Connecting to $host:")

        # Only continue if we're connected
        if isnothing(ssh_state.session)
            continue
        end

        ig.Indent()

        auth_state = ssh_state.auth_state
        auth_method = ssh_state.auth_method

        can_authenticate = false

        if auth_method == ssh.AuthMethod_Password
            ig.Text("Password: ")
            ig.SameLine()
            edited, new_password = SafeInputText("##password"; password=true, max_len=127,
                                                 current_text=ssh_state.password[])
            if edited
                ssh_state.password[] = new_password
            end

            can_authenticate = !isempty(ssh_state.password[])
        elseif auth_method == ssh.AuthMethod_Interactive
            all_answers_filled = true

            for prompt in ssh_state.kbdint_prompts
                ig.Text(prompt.msg)
                ig.SameLine()
                edited, new_answer = SafeInputText("##$prompt"; password=!prompt.display, max_len=127,
                                                   current_text=prompt.answer)
                if edited
                    prompt.answer = new_answer
                end

                if isempty(prompt.answer)
                    all_answers_filled = false
                end
            end

            can_authenticate = all_answers_filled
        elseif ssh_state.auth_state == ssh.KnownHosts_Unknown
            ig.Text("The host is unrecognized, would you like to add it to the known hosts file?")
            ig.SameLine()
            if ig.Button("Yes")
                ssh.update_known_hosts(ssh_state.session)
                @guiasync ssh_authenticate_hop(state[], hop_idx)
                can_authenticate = true
            end
            ig.SameLine()
            ig.Text("/")
            ig.SameLine()
            if ig.Button("No")
                # If they refuse to recognize the host then we can't do anything
                @guiasync disconnect_engine(state[], false)
            end
        else
            can_authenticate = true
        end
        can_authenticate &= auth_state != :authenticating

        ig.Spacing()

        @Disabled !can_authenticate begin
            if ig.Button(auth_state == :authenticating ? "Authenticating" : "Authenticate")
                @guiasync ssh_authenticate_hop(state[], hop_idx)
            end
        end

        if auth_state == :authenticating
            ig.SameLine()
            Spinner()
        elseif auth_state isa ssh.AuthStatus && auth_state != ssh.AuthStatus_Success
            status_str = split(string(auth_state), "_")[2]

            ig.SameLine()
            ig.PushStyleColor(ig.ImGuiCol_Text, ig.IM_COL32(245, 80, 81, 255))
            ig.Text("Error, please try again: '$(status_str)'")
            ig.PopStyleColor()
        end

        ig.Unindent()
    end
end

function restore_plots(state::GuiState)
    client = state.client
    ctx_path = client.context_path
    if !haskey(state.saved_contexts, ctx_path)
        return
    end

    ctx = state.saved_contexts[ctx_path]

    ## Code to restore plots is buggy, so it's disabled for now

    # # Close existing plots before restoring
    # for plot in gui_state.client.plots
    #     close(plot)
    # end
    # empty!(gui_state.client.plots)

    # for p in get(ctx, "plots", [])
    #     dock_id = UInt32(get(p, "dock_id", 0))
    #     if p["type"] == "Plot"
    #         push!(gui_state.client.plots, variable_plot(p["name"], p["id"], dock_id))
    #     else
    #         push!(gui_state.client.plots, correlation_plot(p["id"], dock_id))
    #     end
    # end

    # gui_state.plot_counter = Int(get(ctx, "plot_counter", 0))

    # layout = get(ctx, "saved_layout", "")
    # if !isempty(layout)
    #     # Load the INI to restore dock node topology, then explicitly assign
    #     # each window to its saved dock node. DockBuilderDockWindow stores the
    #     # assignment in ImGui's window settings; it takes effect on the next
    #     # frame once the dock nodes are recreated from the INI.
    #     ig.LoadIniSettingsFromMemory(layout)
    #     # for plot in gui_state.client.plots
    #     #     if plot.dock_id != 0
    #     #         ig.DockBuilderDockWindow(plot.id, plot.dock_id)
    #     #     end
    #     # end
    # end
end

function draw_plots()
    client = state[].client

    # Update all the observables
    updated_variables = Dict{String, Set{Int}}()
    for (name, store) in client.variable_data
        if !isready(store.updates)
            continue
        end

        new_tids = Set{Int}()
        latest_array = nothing  # (tid, payload) — only the last array frame is kept
        while isready(store.updates)
            tid, x, type = take!(store.updates)
            push!(new_tids, tid)
            store.type = type
            if x isa Number
                # Reset scalar_tids too, to preserve the parallel-length
                # invariant — store.data may have been overwritten outside this
                # loop (e.g. ArrayMetadata in client.jl) leaving stale tids.
                if !(store.data isa CircularBuffer)
                    store.data = CircularBuffer{Float64}(SCALAR_BUFFER_CAPACITY)
                    store.scalar_tids = CircularBuffer{Int}(SCALAR_BUFFER_CAPACITY)
                elseif isnothing(store.scalar_tids)
                    store.scalar_tids = CircularBuffer{Int}(SCALAR_BUFFER_CAPACITY)
                end
                push!(store.data, x)
                push!(store.scalar_tids, tid)
            else
                # Arrays are latest-wins: discard intermediate frames and
                # decompress only the most recent one after the loop.
                latest_array = (tid, x)
            end
        end

        if !isnothing(latest_array)
            tid, x = latest_array
            if x isa CompressedArray
                ws = get!(() -> ZfpWorkspace(), client.zfp_workspaces, name)
                # Reuse the existing buffer in place when shape/eltype match,
                # unwrapping any DimArray from a previous frame to get at it.
                prev = store.data
                buf = prev isa DimArray ? parent(prev) : prev
                if buf isa Array && eltype(buf) === x.original_eltype && size(buf) == Tuple(x.shape)
                    decompress_array!(ws, buf, x)
                    store.data = restore_dims(buf, x)
                else
                    store.data = decompress_array(ws, x)
                end
            else
                store.data = x
            end
            store.trainId = tid
            if !isnothing(store.scalar_tids)
                empty!(store.scalar_tids)
            end
        end

        # Update contiguous caches for scalar data so plotting doesn't allocate
        if !isnothing(store.scalar_tids) && store.data isa CircularBuffer
            n = length(store.data)
            resize!(store.scalar_data_cache, n)
            resize!(store.scalar_tids_cache, n)
            copyto!(store.scalar_data_cache, store.data)
            copyto!(store.scalar_tids_cache, store.scalar_tids)
        end

        updated_variables[name] = new_tids
    end

    # Draw plot windows
    for plot in client.plots
        draw_plot(plot, updated_variables)
    end

    # Remove closed plots
    n = length(client.plots)
    for i in reverse(eachindex(client.plots))
        if !client.plots[i].open[]
            close(client.plots[i])
            deleteat!(client.plots, i)
        end
    end
    if length(client.plots) != n
        save_settings(client)
    end
end

function draw_engine_logs()
    client = state[].client

    ig.Dummy(0, 5)
    if ig.Button("Clear all logs")
        empty!(client.engine_logs)
    end

    ig.Dummy(0, 5)
    ig.Separator()

    for (i, log) in enumerate(client.engine_logs)
        ig.PushID(i)

        timestamp = Dates.format(unix2datetime(log.timestamp), client.log_dateformat)
        ig.Text(timestamp)
        ig.SameLine(ig.CalcTextSize("0000-00-00 00:00:00  ").x)

        if !isnothing(log.extra_details)
            CopyButton("log", log.message * "\n" * log.extra_details)
            ig.SameLine()
            if ig.TreeNode(log.message)
                ig.TextUnformatted(log.extra_details)
                ig.TreePop()
            end
        else
            CopyButton("log", log.message)
            ig.SameLine()
            ig.BulletText(log.message)
        end

        ig.PopID()
    end
end

## Main GUI function

default(value, default="") = something(value, default)

function draw_gui()
    # Dock the main window by default
    viewport = ig.igGetMainViewport()
    central_dock_id = ig.DockSpaceOverViewport(ig.GetWindowDockID(), viewport, ig.ImGuiDockNodeFlags_PassthruCentralNode)

    ig.SetNextWindowDockID(central_dock_id, ig.ImGuiCond_Once)

    main_window_flags = ig.ImGuiWindowFlags_NoCollapse
    main_window_flags |= ig.ImGuiWindowFlags_MenuBar
    main_window_flags |= ig.ImGuiWindowFlags_HorizontalScrollbar

    client = state[].client
    fully_authenticated = ssh_fully_authenticated(client)

    # Draw the main window
    if ig.Begin("Main window", C_NULL, main_window_flags)
        # Draw the menubar
        draw_main_menubar()

        if Threads.threadid() == 1
            BorderedText("Warning: GUI is running on thread 1, this may cause performance issues. Start Julia with e.g. `julia -t auto,2` instead.")
        end

        ig.BeginTabBar("main-tab-bar")
        if ig.BeginTabItem("Setup")
            ig.EndTabItem()

            initializing = client.status == RemoteStatus_Initializing
            disconnecting = client.status == RemoteStatus_Disconnecting
            can_connect = if client.embedded_engine
                client.status != RemoteStatus_Connecting && client.status != RemoteStatus_Connected
            else
                client.status != RemoteStatus_Connecting && !fully_authenticated
            end
            can_connect = can_connect && !initializing && !disconnecting

            @Disabled !can_connect begin
                @c ig.Combo("##client-type", &state[].client_type_current_item,
                           ["Connect to remote", "Create local engine"])
                client.embedded_engine = state[].client_type_current_item == 1

                ig.Spacing()

                if !state[].client.embedded_engine
                    ig.Text("Connect to node:")

                    ig.SameLine()

                    edited, new_address = SafeInputText("##client"; hint="exflonc24.desy.de",
                                                        current_text=default(state[].address))

                    ig.Text("Use environment:")
                    ig.SameLine()
                    env_edited, new_environment = SafeInputText("##engine-environment";
                                                                current_text=default(state[].engine_environment))

                    ig.Text("Working directory:")
                    ig.SameLine()
                    wd_edited, new_working_dir = SafeInputText("##engine-working-dir";
                                                               current_text=default(state[].engine_working_dir))

                    if edited
                        state[].address = new_address
                    end
                    if env_edited
                        state[].engine_environment = new_environment
                    end
                    if wd_edited
                        state[].engine_working_dir = new_working_dir
                    end
                end
            end

            ig.Spacing()

            disable_connect = !can_connect || length(state[].address) == 0
            @Disabled disable_connect begin
                if ig.Button("Connect")
                    client.cmd_output = ""
                    client.last_error = ""
                    client.status = RemoteStatus_Initializing
                    @guiasync connect_engine()
                end
            end
            ig.SameLine()

            @Disabled can_connect || initializing || disconnecting begin
                if ig.Button("Disconnect")
                    @guiasync disconnect_engine(state[], false)
                end
            end

            ig.SameLine()

            @Disabled client.status != RemoteStatus_Connected begin
                if ig.Button("Disconnect & shutdown")
                    @guiasync disconnect_engine(state[], true)
                end
            end

            ig.SameLine()

            @Disabled client.status != RemoteStatus_Connected begin
                if ig.Button("Restart")
                    @guiasync restart_engine(state[])
                end
            end

            ig.Dummy(0, 20)
            if client.status == RemoteStatus_Disconnecting
                Spinner("Disconnecting...")
            elseif client.status == RemoteStatus_Initializing
                Spinner("Initializing...")
            elseif client.status == RemoteStatus_Connecting && !fully_authenticated
                draw_ssh_auth()
            elseif client.status == RemoteStatus_Connecting && fully_authenticated
                Spinner("Starting engine...")
                BoxedText("##client_cmd_output", state[].client.cmd_output)
            elseif client.status == RemoteStatus_Error
                @Disabled !fully_authenticated begin
                    if ig.Button("Restart engine")
                        @guiasync initialize_engine(state[])
                    end
                end

                ig.Spacing()

                ig.Text(fully_authenticated ? "Error starting engine:" : "Error connecting to node:")
                BoxedText("##client_last_error", client.last_error)
            elseif client.status == RemoteStatus_Connected
                ig.Text("Connected with client ID: $(client.client_id)")

                ig.Dummy(0, 2)
                ig.Separator()
                ig.Dummy(0, 2)

                @Disabled is_pending(client, client.debug_mode_request) begin
                    ig.Text("Debug mode")
                    ig.SameLine()
                    if ig.Checkbox("##debug-mode", client.debug_mode)
                        set_debug_mode(state[])
                    end

                    if is_pending(client, client.debug_mode_request)
                        ig.SameLine()
                        Spinner()
                    end
                end

                @Disabled client.remoterepl_status == RemoteReplStatus_Changing begin
                    ig.Text("Remote REPL")
                    ig.SameLine()
                    if client.remoterepl_status == RemoteReplStatus_Changing
                        Spinner()
                    elseif ig.Checkbox("##remoterepl-mode", client.remoterepl_mode)
                        set_remoterepl(state[])
                    end
                end

                ig.Text("Use context file:")
                ig.SameLine()
                edited, new_context_path = SafeInputText("##context-file";
                                                         current_text=default(client.context_path))
                if edited
                    client.context_path = new_context_path
                end

                if !client.context_path_valid
                    ig.TextColored(ImVec4(1, 0.2, 0.5, 1), "Path does not point to a valid file!")
                end

                ig.Dummy(0, 10)

                draw_routing_rules()

                ig.Dummy(0, 10)

                @Disabled is_pending(client, client.devices_request) begin
                    if ig.Button("Get devices")
                        get_devices(client)
                    end

                    draw_device_tree(client.device_tree)
                end
            end
        end

        @Disabled client.status != RemoteStatus_Connected || isempty(client.context_path) begin
            if ig.BeginTabItem("Analysis pipeline")
                draw_dag()
                ig.EndTabItem()
            end
        end

        if state[].show_engine_logs
            engine_log_flags = ig.ImGuiTabItemFlags_None
            if state[].select_engine_logs
                engine_log_flags |= ig.ImGuiTabItemFlags_SetSelected
                state[].select_engine_logs = false
            end
            if @c ig.BeginTabItem("Engine logs", &state[].show_engine_logs, engine_log_flags)
                draw_engine_logs()
                ig.EndTabItem()
            end
        end

        ig.EndTabBar()

        draw_plots()

        ig.End()
    end

    # Display tooling windows
    for (window_sym, window_func) in [(:show_imgui_demo,     ig.ShowDemoWindow),
                                      (:show_imgui_metrics,  ig.ShowMetricsWindow),
                                      (:show_stacktool,      ig.ShowIDStackToolWindow),
                                      (:show_debug_log,      ig.ShowDebugLogWindow),
                                      (:show_state_inspector, draw_state_inspector)]
        do_show = getproperty(state[], window_sym)
        if do_show
            @c window_func(&do_show)
            setproperty!(state[], window_sym, do_show)
        end
    end

    # Save layout when ImGui signals changes
    io = ig.GetIO()
    if unsafe_load(io.WantSaveIniSettings) && !isempty(state[].client.plots)
        save_settings(client)
        io.WantSaveIniSettings = false
    end
end

"""Start the XFA GUI."""
function main(; test_engine=nothing)
    # The libXcursor JLL has a build-sandbox icon path baked in, so we need to
    # point it at the system.
    if !haskey(ENV, "XCURSOR_PATH")
        ENV["XCURSOR_PATH"] = "~/.icons:/usr/share/icons:/usr/share/pixmaps"
    end

    gui_state = GuiState(load_settings())

    # Setup Dear ImGui context
    ig.set_backend(:GlfwOpenGL3)
    imgui_ctx = ig.CreateContext()
    ig.SetCurrentContext(imgui_ctx)

    # Setup Dear ImGui style
    ig.StyleColorsDark()
    style = ig.GetStyle()
    style.FrameBorderSize = 1

    # Load fonts
    font_config = ig.ImFontConfig()
    font_config.MergeMode = true
    font_atlas = unsafe_load(ig.GetIO().Fonts)
    font_dir = joinpath(@__DIR__, "fonts")
    fonts=[
        (joinpath(font_dir, "Atkinson_Hyperlegible", "AtkinsonHyperlegible-Regular.ttf"), 20),
        (joinpath(font_dir, "Inter-Regular.otf"), 17),
        (joinpath(font_dir, "JuliaMono-Regular.ttf"), 15),
        (joinpath(font_dir, "JuliaMono-Regular.ttf"), 16),
    ]
    for (font, font_size) in fonts
        ig.AddFontFromFileTTF(font_atlas, font, font_size)
        ig.AddFontFromFileTTF(font_atlas, joinpath(font_dir, "fa-regular-400.otf"), 20, font_config)
        ig.AddFontFromFileTTF(font_atlas, joinpath(font_dir, "fa-solid-900.otf"), 20, font_config)
    end

    # Setup ImPlot context
    implot_ctx = ImPlot.CreateContext()
    add_turbo_colormap()

    # Enable docking and viewports by default
    io = ig.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | ig.ImGuiConfigFlags_DockingEnable
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | ig.ImGuiConfigFlags_ViewportsEnable

    # Disable built-in INI file — we manage layout persistence ourselves
    io.IniFilename = C_NULL

    on_exit = () -> begin
        client = gui_state.client

        # Save layout and node positions before tearing down
        if !isempty(client.plots) || !isempty(client.context.context_state)
            save_settings(client)
        end

        # Disconnect from engine if connected
        if client.status == RemoteStatus_Connected && !isnothing(client.websocket)
            if !WebSockets.isclosed(client.websocket)
                try
                    send(client.websocket, Shutdown())
                    timedwait(() -> WebSockets.isclosed(client.websocket), 5)
                catch
                end
            end
        end

        # Clean up GPU heatmap resources before destroying contexts
        for plot in client.plots
            close(plot)
        end
        destroy_heatmap_context!()

        ImPlot.DestroyContext(implot_ctx)
        if client.ne_editor != C_NULL
            ne.DestroyEditor(client.ne_editor)
        end
        empty!(safe_input_text_cache)
        close(gui_state)
    end

    t = ig.render(imgui_ctx; on_exit, window_title="XFA", wait=false, spawn=true, engine=test_engine) do
        if gui_state.disable_rendering
            # Occasionally an exception will occur in the middle of a disabled
            # section, which helpfully also disables the continue button
            # below. So here we check if we're currently disabled and end it if
            # so.
            if IsItemDisabled()
                ig.igEndDisabled()
            end

            ig.Text("Render loop crashed, continue when ready:")
            @with state => gui_state draw_revise()
            ig.SameLine()
            gui_state.disable_rendering = !ig.Button("Continue")
            return
        end

        try
            @lock gui_state begin
                @with state => gui_state @invokelatest draw_gui()
            end
        catch ex
            @error "Error while rendering:" exception=(ex, catch_backtrace())
            gui_state.disable_rendering = true
        end
    end

    return t, gui_state
end

# precompile(main, ())
# precompile(draw_gui, ())
