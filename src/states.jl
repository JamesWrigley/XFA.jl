@enum RemoteStatus begin
    RemoteStatus_Unconnected
    RemoteStatus_Initializing
    RemoteStatus_Connecting
    RemoteStatus_Connected
    RemoteStatus_Disconnecting
    RemoteStatus_Error
end

@enum RequestStatus begin
    RequestStatus_Idle
    RequestStatus_Waiting
    RequestStatus_Error
end

@enum SshStatus begin
    SshStatus_Unconnected
    SshStatus_Connecting
    SshStatus_NeedsAuth
    SshStatus_Error
end

@enum PipelineStatus begin
    PipelineStatus_Starting
    PipelineStatus_Started
    PipelineStatus_LoadingContext
    PipelineStatus_Stopping
    PipelineStatus_Stopped
end

struct PropertyList
    names::Vector{String}
    displayed_names::Vector{String}
    descriptions::Vector{String}
    value_types::Vector{String}
end
PropertyList() = PropertyList(String[], String[], String[], String[])

struct DeviceProperties
    slow::PropertyList
    fast::Dict{String, PropertyList}
end
DeviceProperties() = DeviceProperties(PropertyList(), Dict{String, PropertyList}())

@enum RemoteReplStatus begin
    RemoteReplStatus_Running
    RemoteReplStatus_Changing
    RemoteReplStatus_Stopped
end

@kwdef mutable struct KaraboDepTextState
    # Working copies of the source/property/proxy fields while the editor window
    # is open. Seeded from the current value when the window opens and composed
    # back into a source string on OK.
    source::String = ""
    property::String = ""
    proxy::String = ""
    proxy_expanded::Bool = false

    # Set by the editor window's OK button (the raw composed source string) and
    # picked up by KaraboDepText on the next frame, which runs the remap and
    # returns the result to the caller.
    committed::Maybe{String} = nothing

    # Per-call context captured when the editor window opens, so the deferred
    # top-level draw has everything it needs. `source_list` is the (possibly
    # topic-filtered) source list to complete against.
    label::String = ""
    source_list::Vector{SourceInfo} = SourceInfo[]
    device_only::Bool = false
    allow_slow::Bool = true
    # One-shot flag to open/focus the window on its first drawn frame.
    trigger::Bool = false
    # Frames the window has been drawn, so the focus-loss close check is skipped
    # on the opening frame.
    frames::Int = 0
    # Set each frame when an autocomplete popup is hovered, so losing window
    # focus to that popup doesn't close the editor.
    ac_hovered::Bool = false
    # Set each frame when an autocomplete popup is open, so an Enter that selects
    # a completion isn't also treated as confirming the whole editor.
    ac_active::Bool = false

    # Set when a new source is picked: the property is re-checked against the new
    # source once its schema arrives, and dropped (with focus moved to the
    # property field) if it no longer belongs.
    revalidate_property::Bool = false
    # One-shot request to focus the property field on the next frame.
    property_focus::Bool = false

    # Set when a remap requires an async device-property lookup. The widget
    # stays disabled until the request resolves, then re-runs the remap.
    # `proxy_property` is filled in by the lookup's callback.
    pending_remap_id::Maybe{Int} = nothing
    pending_remap_source::Maybe{String} = nothing
    proxy_property::Ref{Any} = Ref{Any}(nothing)
end

@kwdef mutable struct DepTextState
    is_karabo::Bool = false
    karabo_state::KaraboDepTextState = KaraboDepTextState()
    # Set when the kind is switched via the (deferred) selector popup so the new
    # text field grabs focus on the next frame.
    wants_focus::Bool = false
end

@kwdef mutable struct CopyableComboPopup
    # The active dropdown request; label is nothing when no popup is open.
    label::Maybe{String} = nothing
    items::Vector{String} = String[]
    anchor::ImVec2 = ImVec2(0, 0)
    width::Cfloat = 0
    # One-shot flag to open the popup on the next deferred draw.
    trigger::Bool = false
    # The chosen selection, held until the widget picks it up next frame.
    result_label::Maybe{String} = nothing
    result_index::Cint = 0
end

@enum ElidedEditState begin
    ElidedEditState_NoEdit
    ElidedEditState_WantEdit
    ElidedEditState_Edit
end

struct CompletionResult
    items::Any
    renderer::Function
    query::String
    source::String
end

@kwdef mutable struct ElidedTextState
    edit::ElidedEditState = ElidedEditState_NoEdit
    selected_idx::Int = 1
    cached_query::String = ""
    cached_source::String = ""
    cached_scored::Vector{Tuple{Int, Any}} = Tuple{Int, Any}[]
    # Source autocomplete only: the row in the channel column of the currently
    # selected device, or 0 when the device row itself is the current row.
    channel_idx::Int = 0
    # Debounce for lazy channel discovery in the source autocomplete: device
    # schemas are only fetched once the query has been stable for a short while.
    last_query::String = ""
    last_change_time::Float64 = 0.0
end

# An autocomplete popup deferred out of the node canvas, like CopyableComboPopup.
# While the canvas is drawing it rewrites the ImGui viewport into canvas-local
# coordinates, and Begin() clips every window against that viewport, so a popup
# positioned in screen space is clipped away to nothing (it is still begun, just
# invisible). Widgets inside the canvas record the request here instead, and
# draw_dag draws it after EndNode under Suspend/Resume.
@kwdef mutable struct CompletionPopup
    # The active request; label is nothing when no widget is completing. It is
    # re-recorded every frame the widget is being edited, and consumed by the
    # deferred draw.
    label::Maybe{String} = nothing
    state::Maybe{ElidedTextState} = nothing
    completions::Maybe{CompletionResult} = nothing
    # Screen-space rect of the input, to hang the popup under.
    input_min::ImVec2 = ImVec2(0, 0)
    input_max::ImVec2 = ImVec2(0, 0)
    # The deferred draw's outcome, keyed by the label it was drawn for and held
    # until that widget picks it up next frame.
    drawn_label::Maybe{String} = nothing
    result::Maybe{String} = nothing
    hovered::Bool = false
end

abstract type AbstractParameterState end

mutable struct OptionalDimsState <: AbstractParameterState
    all_dims::Bool
    pending_text::String
end
OptionalDimsState(param::Parameter{OptionalDims}) = OptionalDimsState(isempty(param.value.dims), "")

mutable struct KbdintPromptState
    msg::String
    display::Bool
    answer::String
end

mutable struct PasswordStore
    const buf::Base.SecretBuffer

    function PasswordStore(password=nothing)
        buf = Base.SecretBuffer()
        if !isnothing(password)
            write(buf, password)
        end

        finalizer(new(buf)) do x
            Base.shred!(x.buf)
        end
    end
end

function Base.getindex(x::PasswordStore)
    str = read(x.buf, String)
    seekstart(x.buf)
    str
end

function Base.setindex!(x::PasswordStore, value::String)
    Base.shred!(x.buf)
    write(x.buf, value)
    seekstart(x.buf)
end

@kwdef mutable struct SshState
    address::String
    port::Int = 22

    auth_state::Any = nothing
    auth_method::Maybe{ssh.AuthMethod} = nothing
    session::Maybe{ssh.Session} = nothing
    forwarder::Maybe{ssh.Forwarder} = nothing

    password::PasswordStore = PasswordStore()
    kbdint_prompts::Vector{KbdintPromptState} = KbdintPromptState[]

    lock::ReentrantLock = ReentrantLock()
end

Base.lock(state::SshState) = lock(state.lock)
Base.unlock(state::SshState) = unlock(state.lock)

function Base.close(state::SshState)
    if !isnothing(state.forwarder)
        close(state.forwarder)
    end
    if !isnothing(state.session)
        close(state.session)
    end

    state.password = PasswordStore()
    empty!(state.kbdint_prompts)
end

# This enum tracks the original type of the variables. We need to distinguish
# this from how they're stored because both scalars and vectors are stored as
# vectors.
@enum VariableType begin
    VariableType_Scalar
    VariableType_Vector
    VariableType_Array
    VariableType_Unknown
end

const SCALAR_BUFFER_CAPACITY = 10_000

@kwdef mutable struct VariableStore
    const updates::Channel = Channel(100)
    data::Union{AbstractArray, CircularBuffer, ArrayMetadata}
    type::VariableType = VariableType_Unknown

    # This field is only used for non-scalar data. Scalar data is stored as a
    # CircularBuffer with a parallel CircularBuffer for train IDs.
    trainId::Int = -1

    # Train IDs for scalar data, parallel to `data` when it's a CircularBuffer
    scalar_tids::Maybe{CircularBuffer{Int}} = nothing

    # Background array decompression. `decode_task` is this variable's in-flight
    # decode (or nothing); while it's running draw_plots drops newer frames
    # instead of spawning another. `decode_tid` is that frame's train ID.
    # `spare_buffer` is the off-screen buffer the task decodes into, swapped with
    # `data` on pickup so decoding never mutates the array being rendered.
    decode_task::Maybe{Task} = nothing
    decode_tid::Int = -1
    spare_buffer::Maybe{Array} = nothing

    # Contiguous caches for plotting scalar CircularBuffer data
    const scalar_data_cache::Vector{Float64} = Float64[]
    const scalar_tids_cache::Vector{Float64} = Float64[]

    # Processing rate (Hz) reported by the engine.
    update_rate::Float64 = 0.0

    # Compression ratio (uncompressed / compressed bytes) of the most recent
    # payload. NaN when the variable arrived uncompressed.
    compression_ratio::Float64 = NaN

    # Size in bytes of the most recent array payload on the wire — the
    # compressed size for compressed payloads, otherwise sizeof(data).
    received_bytes::Int = 0

    # Metadata from VariableData
    title::String = ""
    x_axis::Maybe{AbstractVector} = nothing
    y_axis::Maybe{AbstractVector} = nothing
    xlabel::String = ""
    ylabel::String = ""
    unit::Maybe{String} = nothing
    fixed_aspect::Bool = true
    plot_type::Symbol = :series
    compress::Bool = true
    plot_specs::Vector{PlotSpec} = PlotSpec[]
end

struct LinkInfo
    id::UInt
    start_id::UInt
    end_id::UInt
    channel_key::Tuple{String, String}
end

struct OutputPin
    id::UInt
    label::String
    is_subvariable::Bool
end
OutputPin(id, label) = OutputPin(id, label, false)

# A dependency pin, as needed to rewrite the dependency when a link is dragged
# onto it.
struct DepPinInfo
    node::String
    arg::String
    variable::String
    dep::Dependency
end

# An output pin a dependency can point at
struct OutputPinInfo
    variable::String
    name::String
end

@kwdef mutable struct ContextState
    context_state::Dict{String, Any} = Dict()
    context_path::String = ""
    source::String = ""
    node_positions::Dict{String, Point2d} = Dict()
    pipeline_status::PipelineStatus = PipelineStatus_Stopped

    # The engine's variable-level DAG (variable -> arg -> dependency), used to
    # reject links that would introduce a cycle.
    dag::Dict{String, OrderedDict} = Dict()

    # All parameters in the loaded context, keyed by fully-qualified name.
    # Used by plot overlays to look up @display references without walking
    # context_state. ParameterChanged messages mutate the shared values.
    parameters::Dict{String, Parameter} = Dict()
    # Variable name -> list of parameter names to overlay on its plot.
    displays::Dict{String, Vector{String}} = Dict()

    # Latest per-channel (drops, size, capacity) snapshot from the engine,
    # keyed by (producer, consumer). Updated roughly once per second.
    channel_stats::Dict{Tuple{String, String}, XfaEngine.Context.ChannelStat} = Dict()

    # Latest smoothed Hz at which each input is pushing data, keyed by input
    # name. Updated roughly once per second alongside `channel_stats`.
    input_rates::Dict{String, Float64} = Dict()

    lock::ReentrantLock = ReentrantLock()
end

function ContextState(settings::Dict; kwargs...)
    client_settings = get(settings, "ClientState", Dict{String, Any}())
    context_path = get(client_settings, "context_path", "")

    ContextState(; context_path, kwargs...)
end

Base.lock(ctx::ContextState) = lock(ctx.lock)
Base.unlock(ctx::ContextState) = unlock(ctx.lock)

function Base.setproperty!(ctx::ContextState, sym::Symbol, x)
    @lock ctx setfield!(ctx, sym, x)
end

struct PendingRequest
    msg_type::Type
    sent_at::Float64
end

struct EngineLog
    timestamp::Float64
    message::String
    extra_details::Maybe{String}
end

EngineLog(message::String, extra_details::Maybe{String}=nothing) = EngineLog(time(), message, extra_details)

# Per-variable subscription state. `count` tracks open plots referencing the
# variable; when it drops to zero we flip `active` off (so the engine stops
# streaming) but keep the entry around to remember the user's chosen zfp
# accuracy `k` for the next time a plot of this variable is opened. `k = 0`
# means lossless, `k < 0` lets the engine pick its default, `k > 0` sets the
# fixed-accuracy tolerance.
@kwdef mutable struct SubscriptionState
    count::Int = 0
    k::Float64 = -1
    active::Bool = true
end

# A not-yet-committed node being assembled in the add-variable flow. It's drawn
# as a real node in the main dag-editor (via the shared draw_variable machinery)
# but held here rather than in the context's ctx_state, so its edits stay local
# until the user commits and the source is written. `id` is a synthetic node id
# (unique per pending node, unrelated to any real variable) that all its derived
# pin/attr ids key off. `dep_values` maps each of the spec's dependency args to
# the dependency the user has wired it to.
@kwdef mutable struct PendingNode
    id::UInt
    spec::VariableSpec
    name::String
    dep_values::OrderedDict{String, Dependency} = OrderedDict{String, Dependency}()
    param_values::OrderedDict{Symbol, Parameter} = OrderedDict{Symbol, Parameter}()
    # The var_data dict built for this node this frame (see spec_to_var_data),
    # rebuilt each frame from the fields above.
    var_data::Dict{String, Any} = Dict{String, Any}()
end

@kwdef mutable struct ClientState
    client_id::String = ""
    worker_info::Dict = Dict()

    debug_mode::Ref{Bool} = Ref(false)
    debug_mode_request::Maybe{Int} = nothing
    syncing::Bool = false
    status::RemoteStatus = RemoteStatus_Unconnected
    websocket::Maybe{WebSockets.WebSocket} = nothing
    ssh_hops::Vector{SshState} = SshState[]
    sftp::Maybe{ssh.SftpSession} = nothing
    ws_forwarder::Maybe{ssh.Forwarder} = nothing
    remote_engine_dir::String = ""

    cmd_output::String = ""
    last_error::String = ""

    embedded_engine::Bool = false
    engine::Maybe{EngineState} = nothing

    remoterepl_mode::Ref{Bool} = Ref(false)
    remoterepl_status::RemoteReplStatus = RemoteReplStatus_Stopped

    # Context file and pipeline
    context_path::String = ""
    context_path_valid::Bool = true
    context::ContextState = ContextState()

    available_variables::Vector{VariableSpec} = VariableSpec[]
    # Nodes being assembled in the add-variable flow, not yet written to source.
    pending_nodes::Vector{PendingNode} = PendingNode[]
    # Monotonic counter for minting unique pending-node ids.
    pending_node_counter::UInt = 0

    ne_editor::Ptr{ne.EditorContext} = Ptr{ne.EditorContext}(C_NULL)
    ne_editor_path::String = ""
    ne_node_handles::Dict{UInt, Ptr{ne.NodeId}} = Dict{UInt, Ptr{ne.NodeId}}()
    ne_pin_handles::Dict{UInt, Ptr{ne.PinId}} = Dict{UInt, Ptr{ne.PinId}}()
    ne_link_handles::Dict{UInt, Ptr{ne.LinkId}} = Dict{UInt, Ptr{ne.LinkId}}()
    ne_node_content_widths::Dict{UInt, Float32} = Dict{UInt, Float32}()
    ne_settings::String = ""

    # Pin registries for link dragging, keyed by pin id. Pins that can't take part
    # in a link (the output pins of input nodes, whose Karabo property can't be
    # recovered from the pin) are absent from both.
    ne_dep_pins::Dict{UInt, DepPinInfo} = Dict{UInt, DepPinInfo}()
    ne_output_pins::Dict{UInt, OutputPinInfo} = Dict{UInt, OutputPinInfo}()
    # Dep pins belonging to pending nodes, keyed by pin id -> (node id, arg name).
    # A link accepted onto one of these updates the pending node's dep locally
    # instead of rewriting source. Rebuilt each frame from the pending nodes.
    pending_dep_pins::Dict{UInt, Tuple{UInt, String}} = Dict{UInt, Tuple{UInt, String}}()
    # Scratch handles that QueryNewLink() writes the dragged pin ids into.
    ne_new_link_start::Ptr{ne.PinId} = Ptr{ne.PinId}(C_NULL)
    ne_new_link_end::Ptr{ne.PinId} = Ptr{ne.PinId}(C_NULL)

    # Karabo status
    trainmatchers::Dict{String, Vector{String}} = Dict()
    # The trainmatchers as a source list, which is what Parameter{KaraboDevice}
    # (i.e. an input's trainmatcher) is completed against.
    trainmatcher_sources::Vector{SourceInfo} = SourceInfo[]
    whitelisted_trainmatchers::Set{KaraboDevice} = Set{KaraboDevice}()
    trainmatchers_request_status::RequestStatus = RequestStatus_Idle
    routing_rules::Vector{RoutingRule} = RoutingRule[]
    routing_rules_request_status::RequestStatus = RequestStatus_Idle
    routing_rules_set_request::Maybe{Int} = nothing
    # Effective remap rules supplied by the engine (user rules followed by
    # builtins). Applied to source strings the user enters, in order; every
    # matching rule fires, not just the first.
    remap_rules::Vector{RemapRule} = RemapRule[]
    # Per-row source-autocomplete state for the rules table, keyed by row index.
    routing_rule_source_states::Dict{Int, KaraboDepTextState} = Dict{Int, KaraboDepTextState}()
    # The sources reported by the inputs of the loaded context, sorted by name
    # and flattened across the inputs. This is the only source list the GUI
    # completes Karabo dependencies against.
    source_list::Vector{SourceInfo} = SourceInfo[]
    # source_list grouped by topic. Rebuilt alongside source_list so the routing
    # rules table can look up its per-topic source list without rescanning.
    sources_by_topic::Dict{String, Vector{SourceInfo}} = Dict{String, Vector{SourceInfo}}()

    # Parameter widget states, keyed by parameter name
    parameter_states::Dict{String, AbstractParameterState} = Dict{String, AbstractParameterState}()
    # KaraboDepText widget state, keyed by dependency ID (used for Parameter{KaraboDevice})
    karabo_dep_states::Dict{UInt, KaraboDepTextState} = Dict{UInt, KaraboDepTextState}()
    # DepText widget state, keyed by dependency ID
    dep_text_states::Dict{UInt, DepTextState} = Dict{UInt, DepTextState}()
    dep_kind_popup::Maybe{Tuple{String, DepTextState}} = nothing
    dep_kind_popup_trigger::Bool = false
    # The KaraboDepText editor window currently open, or nothing. Drawn once at
    # the top level of the frame; the inline widgets set this to request it.
    karabo_editor::Maybe{KaraboDepTextState} = nothing
    # CopyableCombo dropdown deferred out of the node canvas. Only one is open at a time.
    combo_popup::CopyableComboPopup = CopyableComboPopup()
    completion_popup::CompletionPopup = CompletionPopup()
    # Variable names available for autocompletion (including subvariable outputs)
    variable_names::Vector{String} = String[]
    source_properties::Dict{Tuple{String, String}, DeviceProperties} = Dict{Tuple{String, String}, DeviceProperties}()
    device_schema_requests::Dict{Tuple{String, String}, Int} = Dict{Tuple{String, String}, Int}()

    # Variables and plots
    variable_data::Dict{String, VariableStore} = Dict()
    variable_gui_states::Dict{String, Any} = Dict()
    plot_counter::Int = 0
    plots::Vector{Plot} = Plot[]

    # Variable subscriptions, keyed by fully-qualified name. Entries are
    # removed when the open-plot count drops to zero. The keys (and each
    # entry's zfp accuracy k) get sent to the engine via SetVariableSubscriptions.
    subscriptions::Dict{String, SubscriptionState} = Dict{String, SubscriptionState}()

    # One zfp workspace per qualified variable name, reused across trains so
    # the decompression scratch buffers don't get resized on every payload.
    zfp_workspaces::Dict{String, ZfpWorkspace} = Dict{String, ZfpWorkspace}()

    # Engine log messages
    engine_logs::Vector{EngineLog} = EngineLog[]
    log_dateformat::Dates.DateFormat = dateformat"yyyy-mm-dd HH:MM:SS"

    # Message tracking
    pending_requests::Dict{Int, PendingRequest} = Dict()
    engine_request_callbacks::Dict{Int, Function} = Dict()

    # Name of the parameter currently being changed; the node graph is disabled
    # while this is set so the user can't queue further changes mid-update.
    # Cleared when the engine echoes a ParameterChanged or replies with an error.
    pending_parameter_change::Maybe{String} = nothing

    # Fully-qualified name of a parameter whose new value should be written
    # back to the context file once the engine confirms the change. Set by the
    # widget that initiated the edit, applied (write file, no reload) on the
    # matching ParameterChanged echo, dropped on error.
    pending_source_edit::Maybe{String} = nothing

    lock::ReentrantLock = ReentrantLock()
end

function ClientState(settings::Dict; kwargs...)
    client_settings = get(settings, "ClientState", Dict{String, Any}())
    context_path = get(client_settings, "context_path", "")
    context = ContextState(settings)

    ClientState(; context_path, context, kwargs...)
end

function Base.lock(state::ClientState)
    lock(state.lock)
    lock(state.context)
end

function Base.unlock(state::ClientState)
    unlock(state.context)
    unlock(state.lock)
end

function Base.setproperty!(state::ClientState, sym::Symbol, x)
    @lock state setfield!(state, sym, x)
    save_settings(state, sym)
end

function Base.show(io::IO, client::ClientState)
    print(io, ClientState, "(client_id=$(client.client_id), $(client.status), $(length(client.ssh_hops)) SSH hops)")
end

function Base.close(client::ClientState)
    if !isnothing(client.websocket)
        close(client.websocket)
    end

    # Kill the SSH connections
    if !isnothing(client.sftp)
        close(client.sftp)
    end
    if !isnothing(client.ws_forwarder)
        close(client.ws_forwarder)
    end

    for ssh_state in Iterators.reverse(client.ssh_hops)
        close(ssh_state)
    end

    # Kill any local engine
    if !isnothing(client.engine)
        notify(client.engine.stop_event)
        wait(client.engine.stop_task)
    end

    # Delete any cached values
    empty!(safe_input_text_cache)
    empty!(client.variable_data)
end

# A baked soft drop-shadow texture plus the 9-slice geometry to draw it around a
# window. Built lazily (needs a live GL context); see build_window_shadow!.
struct WindowShadow
    tex::GLuint
    size::Int
    cell::Int
    margin::Float32
    corner::Float32
end

@kwdef mutable struct GuiState
    disable_rendering::Bool = false

    # Baked drop-shadow for the Karabo source editor, built on first use.
    window_shadow::Maybe{WindowShadow} = nothing

    # Showing external tool windows
    show_imgui_demo::Bool = false
    show_imgui_metrics::Bool = false
    show_implot_metrics::Bool = false
    show_implot_demo::Bool = false
    show_stacktool::Bool = false
    show_debug_log::Bool = false
    show_state_inspector::Bool = false
    show_engine_logs::Bool = false
    select_engine_logs::Bool = false

    # Connections to remote things
    address::String = "wrigleyj@exflonc202.desy.de"
    client::ClientState = ClientState()
    engine_environment::String = "@xfa-default"
    engine_working_dir::String = "/scratch/xfa"
    client_type_current_item::Cint = Cint(0)

    # Plot layout persistence, keyed by context path
    saved_contexts::Dict{String, Dict{String, Any}} = Dict()

    lock::ReentrantLock = ReentrantLock()
end

function GuiState(settings::Dict; kwargs...)
    gui = get(settings, "GuiState", Dict{String, Any}())
    client_settings = get(settings, "ClientState", Dict{String, Any}())

    address = get(gui, "address", "wrigleyj@exflonc202.desy.de")
    engine_environment = get(gui, "engine_environment", "@xfa-default")
    engine_working_dir = get(gui, "engine_working_dir", "/scratch/xfa")
    client_type_current_item = Cint(get(gui, "client_type", 0))
    saved_contexts = Dict{String, Dict{String, Any}}(
        k => Dict{String, Any}(v) for (k, v)
            in get(client_settings, "contexts", Dict()))

    client = ClientState(settings; embedded_engine = client_type_current_item == 1)

    GuiState(; address, engine_environment, engine_working_dir,
             client_type_current_item, saved_contexts, client, kwargs...)
end

function Base.setproperty!(state::GuiState, sym::Symbol, x)
    @lock state setfield!(state, sym, x)
    save_settings(state, sym)
end

function Base.show(io::IO, state::GuiState)
    print(io, GuiState, "(engine_environment=\"$(state.engine_environment)\")")
end

function Base.close(state::GuiState)
    close(state.client)
end

function Base.lock(state::GuiState)
    lock(state.lock)
    lock(state.client.lock)
end

function Base.unlock(state::GuiState)
    unlock(state.client.lock)
    unlock(state.lock)
end
