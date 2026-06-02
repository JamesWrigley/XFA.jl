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

const SourceInfo = @NamedTuple{topic::String, name::String, ambiguous::Bool}

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
    cursor_pos::Cint = -1
    device::Maybe{String} = nothing

    # If set, the callback will replace the buffer contents with this text,
    # move the cursor to the end, and then clear it.
    wanted_text::Maybe{String} = nothing

    # Set when a remap requires an async device-property lookup. The widget
    # stays disabled until the request resolves, then re-runs the remap.
    # `proxy_property` is filled in by the lookup's callback.
    pending_remap_id::Maybe{Int} = nothing
    pending_remap_source::Maybe{String} = nothing
    proxy_property::Ref{Any} = Ref{Any}(nothing)
end

mutable struct DepTextState
    is_karabo::Bool
    karabo_state::KaraboDepTextState
end

DepTextState(is_karabo::Bool) = DepTextState(is_karabo, KaraboDepTextState())

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
    id::Cint
    start_id::Cint
    end_id::Cint
    channel_key::Tuple{String, String}
end

struct OutputPin
    id::Cint
    label::String
    is_subvariable::Bool
end
OutputPin(id, label) = OutputPin(id, label, false)

@kwdef mutable struct ContextState
    context_state::Dict{String, Any} = Dict()
    context_path::String = ""
    source::String = ""
    node_positions::Dict{String, Point2d} = Dict()
    pipeline_status::PipelineStatus = PipelineStatus_Stopped

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

    node_positions = Dict{String, Point2d}()
    contexts = get(client_settings, "contexts", Dict())
    if haskey(contexts, context_path)
        saved_positions = get(contexts[context_path], "node_positions", Dict())
        for (name, pos) in saved_positions
            node_positions[name] = Point2d(pos[1], pos[2])
        end
    end

    ContextState(; context_path, node_positions, kwargs...)
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
# streaming) but keep the entry around to remember the user's chosen
# `precision` for the next time a plot of this variable is opened.
@kwdef mutable struct SubscriptionState
    count::Int = 0
    precision::Int = -1
    active::Bool = true
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

    webproxy_status::RequestStatus = RequestStatus_Idle
    remoterepl_mode::Ref{Bool} = Ref(false)
    remoterepl_status::RemoteReplStatus = RemoteReplStatus_Stopped

    # Context file and pipeline
    context_path::String = ""
    context_path_valid::Bool = true
    context::ContextState = ContextState()

    # Karabo status
    trainmatchers::Dict{String, Vector{String}} = Dict()
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
    karabo_devices::Dict{String, Dict{String, Any}} = Dict()
    devices_request::Maybe{Int} = nothing
    # Pre-sorted for display: [(topic, [(device_name, sorted_info_pairs), ...]), ...]
    device_tree::Vector{Tuple{String, Vector{Tuple{String, Vector{Pair{String, Any}}}}}} = []
    # Flat list of sources for autocompletion. Sources include both devices and
    # their pipeline outputs (e.g. "foo" and "foo:output"). The ambiguous flag
    # indicates that the source name appears in more than one topic.
    source_list::Vector{SourceInfo} = SourceInfo[]
    # source_list grouped by topic. Rebuilt alongside source_list so the routing
    # rules table can look up its per-topic source list without rescanning.
    sources_by_topic::Dict{String, Vector{SourceInfo}} = Dict{String, Vector{SourceInfo}}()

    # Parameter widget states, keyed by parameter name
    parameter_states::Dict{String, AbstractParameterState} = Dict{String, AbstractParameterState}()
    # KaraboDepText widget state, keyed by dependency ID (used for Parameter{KaraboDevice})
    karabo_dep_states::Dict{Int, KaraboDepTextState} = Dict{Int, KaraboDepTextState}()
    # DepText widget state, keyed by dependency ID
    dep_text_states::Dict{Int, DepTextState} = Dict{Int, DepTextState}()
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
    # entry's precision) get sent to the engine via SetVariableSubscriptions.
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

@kwdef mutable struct GuiState
    disable_rendering::Bool = false

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
