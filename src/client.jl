const BASTION = "bastion.desy.de"
const GATEWAY = "exflgateway.desy.de"

function log_engine_error(state::GuiState, message::String, extra_details::Maybe{String}=nothing)
    push!(state.client.engine_logs, EngineLog(message, extra_details))
    state.show_engine_logs = true
    state.select_engine_logs = true
end

function send(client::ClientState, msg::AbstractMessage)
    id = Protocol.client_send(client.websocket, msg)
    client.pending_requests[id] = PendingRequest(typeof(msg), time())
    return id
end

function is_pending(client::ClientState, id::Union{MessageId, Nothing})
    !isnothing(id) && haskey(client.pending_requests, id)
end

# Send a request to the engine with a callback that will be executed when the
# response arrives. Returns the message ID.
function send_with_callback(client::ClientState, msg::AbstractMessage, callback::Function)
    id = send(client, msg)
    client.engine_request_callbacks[id] = callback
    return id
end

function peekall(buffer::IOBuffer)
    return String(take!(copy(buffer)))
end

# Single ping with a 1s timeout; used to detect whether we can skip bastion/gateway jumps.
function is_directly_reachable(address::AbstractString)
    success(pipeline(`ping -c 1 -W 1 $address`; stdout=devnull, stderr=devnull))
end

function ssh_initialize(state::GuiState)
    client = state.client
    client.status = RemoteStatus_Connecting
    address = state.address
    user = nothing
    if occursin("@", address)
        user, address = split(address, "@")
    end

    if endswith(address, ".desy.de") && address != GATEWAY && address != BASTION && !is_directly_reachable(address)
        push!(client.ssh_hops, SshState(; address=BASTION))
        push!(client.ssh_hops, SshState(; address=GATEWAY))
    end

    # This is the blocking SSH session used for SFTP
    push!(client.ssh_hops, SshState(; address))
    # And this is the regular, non-blocking SSH session
    push!(client.ssh_hops, SshState(; address))

    # Start by initializing the first hop
    ssh_initialize_hop(state, 1, user)
end

# Pin every SSH session's actor + fd poller to one dedicated default-pool
# thread so their per-poll notify/wait handshake stays thread-local instead of
# thrashing the cross-thread scheduler. The last default thread tends to be
# quieter than the first, which GUI/engine work lands on.
ssh_pin_tid() = first(Threads.threadpooltids(:default))

function ssh_initialize_hop(state, hop_idx, user)
    client = state.client
    ssh_state = client.ssh_hops[hop_idx]
    forwarder_idx = findlast(x -> !isnothing(x.forwarder), client.ssh_hops)

    if !isnothing(forwarder_idx) # hop_idx > firstindex(client.ssh_hops)
        # Connect to the forwarded port 22
        forwarder = client.ssh_hops[forwarder_idx].forwarder
        session = ssh.Session(forwarder.localinterface, forwarder.localport;
                               user, pin_tid=ssh_pin_tid())

        # Reset the host so that GSSAPI auth works
        session.host = ssh_state.address

        ssh_state.session = session
    else
        ssh_state.session = ssh.Session(ssh_state.address, ssh_state.port;
                                        user, pin_tid=ssh_pin_tid())
    end

    ssh_authenticate_hop(state, hop_idx)
end

function ssh_fully_authenticated(client::ClientState)
    hops = client.ssh_hops
    return !isempty(hops) && all([hop.auth_state == ssh.AuthStatus_Success for hop in hops])
end

function ssh_authenticate_hop(state::GuiState, hop_idx)
    client = state.client
    ssh_state = client.ssh_hops[hop_idx]
    session = ssh_state.session
    auth_method = ssh_state.auth_method

    ssh_state.auth_state = :authenticating

    new_auth_state = if auth_method == ssh.AuthMethod_Password
        ssh.authenticate(session; password=ssh_state.password[], throw=false)
    elseif auth_method == ssh.AuthMethod_Interactive
        kbdint_answers = [prompt.answer for prompt in ssh_state.kbdint_prompts]

        ssh.authenticate(session; kbdint_answers, throw=false)
    else
        ssh.authenticate(session; throw=false)
    end

    # If we're doing interactive auth and the server asks more questions then we
    # update the prompts.
    if new_auth_state == ssh.AuthMethod_Interactive
        update_auth_prompts(ssh_state)
    end

    # At this point we're done handling the response from the server so update
    # the state for the GUI.
    if new_auth_state isa ssh.AuthMethod
        ssh_state.auth_method = new_auth_state
        ssh_state.auth_state = nothing
    elseif new_auth_state isa ssh.AuthStatus
        ssh_state.auth_state = new_auth_state
    elseif new_auth_state isa ssh.KnownHosts
        ssh_state.auth_state = new_auth_state
    else
        @error "Unsupported result from ssh.authenticate(): $(new_auth_state)"
    end

    if new_auth_state == ssh.AuthStatus_Success
        last_idx = lastindex(client.ssh_hops)
        sftp_idx = last_idx - 1

        if hop_idx == last_idx
            # If we're the last hop in the SSH chain and authentication succeeded, start
            # the engine too.
            initialize_engine(state)
        else
            next_hop = client.ssh_hops[hop_idx + 1]
            next_hop.auth_state = :connecting

            if hop_idx < sftp_idx
                # If we haven't gotten to the final node yet, create a forwarder
                localport = getavailableport(1332; interface=Sockets.localhost)
                ssh_state.forwarder = ssh.Forwarder(session, localport,
                                                    next_hop.address, next_hop.port;
                                                    localinterface=Sockets.localhost)
            elseif hop_idx == sftp_idx
                # If we're at the SFTP hop then we can create an SFTP session
                client.sftp = ssh.SftpSession(session)
            end

            ssh_initialize_hop(state, hop_idx + 1, session.user)
        end
    end
end

function update_auth_prompts(ssh_state)
    session = ssh_state.session
    prompts = ssh.userauth_kbdint_getprompts(session)

    ssh_state.kbdint_prompts = [KbdintPromptState(prompt.msg, prompt.display, "") for prompt in prompts]
end

function auth_supported(auth_method)
    if auth_method in (ssh.AuthMethod_Password, ssh.AuthMethod_Interactive)
        true
    elseif auth_method == ssh.AuthMethod_GSSAPI_MIC
        ssh.Gssapi.isavailable()
    else
        false
    end
end

function sync_files()
    client = state[].client
    if client.embedded_engine
        return
    end

    client.syncing = true
    try
        engine_dir = joinpath(pkgdir(XfaEngine), "src")
        git_diff = readchomp(`git diff --name-only $(engine_dir)`)
        if isempty(git_diff)
            @info "No files to sync"
            return
        end

        changed_files = split(git_diff, "\n")

        for path in changed_files
            local_path = joinpath(engine_dir, basename(path))
            remote_path = joinpath(client.remote_engine_dir, "src", basename(path))

            # Note that we read `local_path` before opening `remote_path`. This to
            # avoid the file getting truncated by `open(; write=true)` if we're
            # SSH'ing locally.
            data = read(local_path)
            open(remote_path, client.sftp; write=true) do f
                write(f, data)
            end
        end
    finally
        client.syncing = false
    end
end


function initialize_engine(state)
    client = state.client
    client.status = RemoteStatus_Connecting
    is_local = client.embedded_engine || state.address == "localhost"

    julia_module_prefix = if is_local
        "true"
    else
        "source /etc/profile.d/modules.sh; SASE=0 module load exfel julia/202602 > /dev/null 2>&1"
    end

    bootstrap_process = nothing

    try
        if client.embedded_engine
            client.engine = XfaEngine.main(; wait=false)
        else
            address = state.address
            ssh_state = client.ssh_hops[end]
            session = ssh_state.session

            working_dir = is_local ? pwd() : state.engine_working_dir
            bootstrap_jl = joinpath(working_dir, "bootstrap.jl")
            code = read(joinpath(@__DIR__, "bootstrap.jl"))

            mkpath(dirname(bootstrap_jl), client.sftp)
            open(bootstrap_jl, client.sftp; write=true) do f
                write(f, code)
            end

            bootstrap_env = Dict("XFA_WORKING_DIR" => working_dir)
            bootstrap_env_str = join(["$(key)=$(value)" for (key, value) in bootstrap_env], " ")
            bootstrap_cmd = "$(bootstrap_env_str) bash -c '$(julia_module_prefix); julia --project=$(state.engine_environment) --color=no $(bootstrap_jl)'"
            bootstrap_process = run(bootstrap_cmd, session; wait=false)

            while !process_exited(bootstrap_process)
                client.cmd_output = String(copy(bootstrap_process.out))
                sleep(0.5)
            end
            client.cmd_output = String(copy(bootstrap_process.out))

            # Read the worker info
            output = client.cmd_output
            find_start = findfirst(">>>", output)
            if isnothing(find_start)
                error("Empty output from bootstrap command")
            end
            find_end = findfirst("<<<", output)
            if isnothing(find_end)
                error("Couldn't read worker info from bootstrap command")
            end

            toml_str = output[find_start.stop + 1:find_end.start - 1]
            worker_info = TOML.parse(toml_str)
            client.worker_info = worker_info

            ws_port = worker_info["1"]["websocket-port"]
            bridge_port = worker_info["1"]["karabo-bridge-port"]

            local_ws_port = getavailableport(ws_port; interface=Sockets.localhost)
            client.ws_forwarder = ssh.Forwarder(session, local_ws_port, ssh_state.address, ws_port;
                                                localinterface=Sockets.localhost)
        end

        @guiasync handle_server(state)
    catch ex
        output_str = if isnothing(bootstrap_process)
            ""
        else
            "Command output:\n" * String(copy(bootstrap_process.out)) * "\n\n"
        end

        backtrace = Util.exception2str(ex, catch_backtrace())
        full_error = output_str * "Exception:\n" * backtrace

        client.last_error = full_error
        client.status = RemoteStatus_Error
    end
end

function connect_engine()
    client = state[].client
    if client.embedded_engine
        initialize_engine(state[])
    else
        ssh_initialize(state[])
    end
end

function disconnect_engine(state, shutdown_engine)
    client = state.client
    client.status = RemoteStatus_Disconnecting

    if shutdown_engine && !isnothing(client.websocket)
        if !WebSockets.isclosed(client.websocket)
            send(client, Shutdown())
        end

        # Wait for the websocket to be closed before killing the connection
        timedwait(() -> WebSockets.isclosed(client.websocket), 10)
    end

    if client.embedded_engine
        rm("worker-info.toml"; force=true)
    end

    close(state.client)
    # Note that we use setfield!() here to bypass the locking, which would
    # otherwise cause locking mismatches.
    setfield!(state, :client, ClientState(load_settings()))
end

# Restart the engine by shutting it down, waiting for it to die, and then
# re-initializing it. The SSH session is kept alive.
function restart_engine(state)
    client = state.client
    if client.status ∉ (RemoteStatus_Connected, RemoteStatus_Initializing)
        @warn "The engine has status $(client.status), it's not valid to restart in that state"
        return
    end

    client.status = RemoteStatus_Disconnecting

    # Send the shutdown message
    if !isnothing(client.websocket) && !WebSockets.isclosed(client.websocket)
        send(client, Shutdown())
        timedwait(() -> WebSockets.isclosed(client.websocket), 10)
    end

    # Wait for the engine process to die
    pid = Int(client.worker_info["1"]["pid"])
    if client.embedded_engine
        if !isnothing(client.engine)
            notify(client.engine.stop_event)
            wait(client.engine.stop_task)
        end
        rm("worker-info.toml"; force=true)
    else
        session = client.ssh_hops[end].session
        run("for i in \$(seq 1 20); do kill -0 $(pid) 2>/dev/null || break; sleep 0.5; done", session; wait=true)
    end

    # Close the websocket and forwarder but keep SSH alive
    if !isnothing(client.websocket)
        close(client.websocket)
    end
    if !isnothing(client.ws_forwarder)
        close(client.ws_forwarder)
    end

    client.websocket = nothing
    client.ws_forwarder = nothing
    client.worker_info = Dict()
    client.engine = nothing
    empty!(client.variable_data)

    # Re-initialize the engine
    initialize_engine(state)
end

# 32-bit hash for ImPlot drag-tool ids, whose id parameter is a C int. Node
# editor ids use hash() directly (a Csize_t-width, content-derived id).
int32_hash(x, y) = reinterpret(Cint, crc32c(y, crc32c(x)))

function build_context_state(state, ctx_info)
    ctx_state = Dict{String, Any}()
    # Drop all per-widget editing state so freshly-loaded parameter values
    # overwrite anything the user had typed but not committed.
    empty!(state.client.parameter_states)
    empty!(state.client.karabo_dep_states)
    empty!(state.client.dep_text_states)
    empty!(state.client.ne_dep_pins)
    empty!(state.client.ne_output_pins)
    empty!(state.client.pending_nodes)
    empty!(safe_input_text_cache)

    group_names = Set(ctx_info["groups"])
    is_group_var(name) = any(startswith(name, "$(g).") for g in group_names)
    group_of(name) = first(g for g in group_names if startswith(name, "$(g)."))

    postprocessors_info = get(ctx_info, "postprocessors", Dict{String, Vector{String}}())

    # Build regular (non-group) variable nodes
    for (name, deps) in ctx_info["dag"]
        if is_group_var(name)
            continue
        end

        ctx_state[name] = Dict{String, Any}("id" => hash(name))

        ctx_state[name]["dependencies"] = []
        ctx_state[name]["outputs"] = []
        ctx_state[name]["postprocessors"] = []
        ctx_state[name]["type"] = :variable
        ctx_state[name]["origin"] = ctx_info["origins"][name]
        ctx_state[name]["draw_parameters"] = true

        for (arg_name, dep) in deps
            attr_id = hash("$(name).dependencies.$(arg_name => dep)")
            push!(ctx_state[name]["dependencies"], DependencyPin(; id=attr_id, arg_name, dep))
            if dep isa Dependency
                state.client.ne_dep_pins[attr_id] = DepPinInfo(name, arg_name, name, dep)
            end
        end

        # The variable itself is always the first output
        output_id = hash("$(name).outputs.")
        push!(ctx_state[name]["outputs"], OutputPin(output_id, ""))
        state.client.ne_output_pins[output_id] = OutputPinInfo(name, name)

        pp_names = Set(get(postprocessors_info, name, String[]))
        for subvar in ctx_info["subvariables"][name]
            subvar_id = hash("$(name).outputs.$(subvar)")
            state.client.ne_output_pins[subvar_id] = OutputPinInfo(name, subvar)
            if subvar in pp_names
                pp_prefix = "$(subvar)."
                pp_params = Dict{String, Any}()
                for (param_name, param) in ctx_info["parameters"]
                    if startswith(param_name, pp_prefix)
                        pp_params[chopprefix(param_name, pp_prefix)] = param
                    end
                end
                push!(ctx_state[name]["postprocessors"], (
                    id = subvar_id,
                    name = subvar,
                    display_name = chopprefix(subvar, "$(name)."),
                    tree_id_suffix = "###pp_$(subvar)",
                    plot_id = "Plot##pp_plot_$(subvar)",
                    params = pp_params,
                    origin = ctx_info["postprocessor_origins"][subvar],
                ))
            else
                push!(ctx_state[name]["outputs"], OutputPin(subvar_id, chopprefix(subvar, "$(name)."), true))
            end
        end
    end

    # Build group nodes with member variables folded in as inputs/outputs
    for name in ctx_info["groups"]
        group_filter = startswith("$(name).")

        ctx_state[name] = Dict{String, Any}("id" => hash(name))
        ctx_state[name]["dependencies"] = []
        ctx_state[name]["outputs"] = []
        ctx_state[name]["type"] = :group
        ctx_state[name]["origin"] = ctx_info["origins"][name]
        ctx_state[name]["draw_parameters"] = true
        ctx_state[name]["links"] = LinkInfo[]
        ctx_state[name]["parameters"] = Dict{String, Any}()

        group_param_args = get(ctx_info, "group_parameter_args", Dict{String, Dict{String, String}}())

        # Add Parameter{Dependency} fields of the group struct as inputs. Member
        # variable dependencies that don't bind to a group field are excluded:
        # they're either group-internal references or static (and editable via
        # parameters).
        dep_param_names = Set{String}()
        seen_field_names = Set{String}()
        for (var_name, deps) in ctx_info["dag"]
            if !group_filter(var_name)
                continue
            end

            for (arg_name, dep) in deps
                field_name = get(get(group_param_args, var_name, Dict{String, String}()), arg_name, nothing)
                if isnothing(field_name)
                    continue
                end
                if field_name in seen_field_names
                    continue
                end
                push!(seen_field_names, field_name)
                attr_id = hash("$(var_name).dependencies.$(arg_name => dep)")
                push!(ctx_state[name]["dependencies"],
                      DependencyPin(; id=attr_id, arg_name, dep, field=field_name,
                                    optional=ctx_info["parameters"]["$(name).$(field_name)"].optional))
                push!(dep_param_names, field_name)
                # The group's kwarg is what gets rewritten, not the member
                # variable's argument.
                if dep isa Dependency
                    state.client.ne_dep_pins[attr_id] = DepPinInfo(name, field_name, var_name, dep)
                end
            end
        end

        # Add group inputs as outputs
        inputs = filter(group_filter, keys(ctx_info["inputs"]))
        for input_name in inputs
            stripped_name = chopprefix(input_name, "$(name).")
            push!(ctx_state[name]["outputs"], OutputPin(hash(input_name), stripped_name))
        end

        # Add group variables from the DAG as outputs
        for (var_name, _) in ctx_info["dag"]
            if !group_filter(var_name)
                continue
            end
            stripped_name = chopprefix(var_name, "$(name).")

            # The variable itself
            attr_id = hash("$(var_name).outputs.")
            push!(ctx_state[name]["outputs"], OutputPin(attr_id, stripped_name))
            state.client.ne_output_pins[attr_id] = OutputPinInfo(var_name, var_name)

            # Its subvariables
            for subvar in ctx_info["subvariables"][var_name]
                subvar_id = hash("$(var_name).outputs.$(subvar)")
                push!(ctx_state[name]["outputs"], OutputPin(subvar_id, chopprefix(subvar, "$(name)."), true))
                state.client.ne_output_pins[subvar_id] = OutputPinInfo(var_name, subvar)
            end
        end

        for (param_name, param) in ctx_info["parameters"]
            if !group_filter(param_name)
                continue
            end

            stripped_name = chopprefix(param_name, "$(name).")
            if stripped_name in dep_param_names
                # Already shown as a dependency input
                continue
            end

            if param isa Parameter{Dependency} && param.optional
                # An unwired optional dependency is absent from the DAG, but it
                # still gets a pin so a link can be dragged onto it.
                attr_id = hash("$(name).dependencies.$(stripped_name)")
                dep = Dependency("")
                # The member variable consuming the field, for the cycle check
                # when a link is dropped on the pin.
                owner = name
                for (var_name, mapping) in group_param_args
                    if group_filter(var_name) && stripped_name in values(mapping)
                        owner = var_name
                        break
                    end
                end
                push!(ctx_state[name]["dependencies"],
                      DependencyPin(; id=attr_id, arg_name=stripped_name, dep, field=stripped_name, optional=true))
                state.client.ne_dep_pins[attr_id] = DepPinInfo(name, stripped_name, owner, dep)
            else
                ctx_state[name]["parameters"][stripped_name] = param
            end
        end
    end

    for name in keys(ctx_info["inputs"])
        # Inputs that are part of groups will have been added before
        if any(startswith(name, "$(group).") for group in ctx_info["groups"])
            continue
        end

        if !haskey(ctx_info, name)
            ctx_state[name] = Dict{String, Any}("id" => hash(name))
            ctx_state[name]["dependencies"] = []
            ctx_state[name]["outputs"] = [OutputPin(hash(name), name)]
            ctx_state[name]["type"] = :input
            ctx_state[name]["links"] = LinkInfo[]
        end
    end

    node_dag = Dict(name => String[] for name in keys(ctx_state))

    # Reverse lookup: (group_name, field_name) -> the single attr_id used as the
    # group's pin for that field. Multiple member-variable deps bound to the
    # same group field collapse to one link.
    group_field_pins = Dict{Tuple{String, String}, UInt}()
    for gname in ctx_info["groups"]
        for pin in ctx_state[gname]["dependencies"]
            group_field_pins[(gname, pin.field)] = pin.id
        end
    end
    group_param_args_all = get(ctx_info, "group_parameter_args", Dict{String, Dict{String, String}}())

    new_links = LinkInfo[]
    seen_link_ids = Set{UInt}()
    for (name, deps) in ctx_info["dag"]
        # Determine which node this variable belongs to
        node_name = is_group_var(name) ? group_of(name) : name

        for (arg_name, dep) in deps
            # Skip deps that weren't added as pins
            if is_group_var(name) && dep isa Dependency && dep.kind == DepKind_Group
                continue
            end

            link_end_id = if is_group_var(name)
                field_name = get(get(group_param_args_all, name, Dict{String, String}()), arg_name, nothing)
                isnothing(field_name) ? nothing : get(group_field_pins, (node_name, field_name), nothing)
            else
                hash("$(name).dependencies.$(arg_name => dep)")
            end
            if isnothing(link_end_id)
                continue
            end

            if dep isa Dependency && dep.kind in (DepKind_Variable, DepKind_Subvariable)
                owner, pin_suffix = dep.kind == DepKind_Variable ? (dep.name, "") : (dep.parent, dep.name)
                link_start_id = hash("$(owner).outputs.$(pin_suffix)")
                dep_node = is_group_var(owner) ? group_of(owner) : owner
                link_id = hash("$(link_start_id)->$(link_end_id)")
                if !(link_id in seen_link_ids)
                    push!(seen_link_ids, link_id)
                    push!(new_links, LinkInfo(link_id, link_start_id, link_end_id, (dep.name, name)))
                end

                if dep_node != node_name
                    push!(node_dag[node_name], dep_node)
                end
            elseif dep isa Dependency && dep.kind == DepKind_Karabo
                input_name = ctx_info["dep_to_input"][dep.name]
                link_start_id = hash(input_name)
                link_id = hash("$(link_start_id)->$(link_end_id)")
                if !(link_id in seen_link_ids)
                    push!(seen_link_ids, link_id)
                    push!(new_links, LinkInfo(link_id, link_start_id, link_end_id, (dep.name, name)))
                end

                input_node_name = split(input_name, ".")[1]
                if input_node_name != node_name
                    push!(node_dag[node_name], input_node_name)
                end
            end
        end

        ctx_state[node_name]["links"] = new_links
    end

    # positions = NetworkLayout.squaregrid(adj_matrix) .* 200
    # positions[node2index[name]]
    new_positions = Dict{String, Point2d}()
    levels = coffman_graham(node_dag)
    for (level, nodes) in levels
        x_pos = level * 400
        for (i, node) in enumerate(nodes)
            new_positions[node] = Point2d(x_pos + i * 300, i * 200)
        end
    end

    state.client.context.node_positions = merge(new_positions, state.client.context.node_positions)

    for var_data in values(ctx_state)
        var_data["renaming"] = false
    end

    # Build variable names list for DepText autocompletion
    var_names = String[]
    for (name, _) in ctx_info["dag"]
        push!(var_names, name)
        for subvar in ctx_info["subvariables"][name]
            push!(var_names, subvar)
        end
    end
    sort!(var_names)
    state.client.variable_names = var_names

    for (param_name, param) in ctx_info["parameters"]
        if param isa Parameter{OptionalDims}
            state.client.parameter_states[param_name] = OptionalDimsState(param)
        end
    end

    # Mirror the engine's parameter dict and @display map onto the client
    # context so plot overlays can look up referenced parameters by name.
    state.client.context.dag = ctx_info["dag"]
    state.client.context.parameters = Dict{String, Parameter}(ctx_info["parameters"])
    state.client.context.displays = Dict{String, Vector{String}}(get(ctx_info, "displays", Dict{String, Vector{String}}()))

    return ctx_state
end

function coffman_graham(dag; W=3)
    order = XfaEngine.Context.topological_sort(dag)
    levels = Dict(0 => String[])
    current_level = 0
    for node in order
        if length(levels[current_level]) < W
            push!(levels[current_level], node)
        else
            current_level += 1
            levels[current_level] = String[node]
        end
    end

    return levels
end

function schema_property_names(schema::AbstractDict)
    props = DeviceProperties()
    collect_properties!(props, "", schema)
    slow_order = sortperm(props.slow.names)
    sorted_slow = PropertyList(props.slow.names[slow_order], props.slow.displayed_names[slow_order],
                               props.slow.descriptions[slow_order], props.slow.value_types[slow_order])
    sorted_fast = Dict{String, PropertyList}()
    for (pipeline, plist) in props.fast
        order = sortperm(plist.names)
        sorted_fast[pipeline] = PropertyList(plist.names[order], plist.displayed_names[order],
                                             plist.descriptions[order], plist.value_types[order])
    end
    return DeviceProperties(sorted_slow, sorted_fast)
end

const NDARRAY_PROPERTIES = ("data", "shape", "type", "isBigEndian")

function collect_properties!(props, prefix, node::AbstractDict, target::PropertyList=props.slow)
    for (key, value) in node
        path = isempty(prefix) ? key : "$(prefix).$(key)"
        if value isa AbstractDict
            if get(value, "nodeType", "") == "Leaf"
                push!(target.names, path)
                push!(target.displayed_names, get(value, "displayedName", ""))
                push!(target.descriptions, get(value, "description", ""))
                push!(target.value_types, get(value, "valueType", ""))
            elseif haskey(value, "noInputShared") && haskey(value, "schema")
                pipeline_props = get!(props.fast, path, PropertyList())
                collect_properties!(props, "", value["schema"], pipeline_props)
            elseif issetequal(keys(value), NDARRAY_PROPERTIES)
                push!(target.names, path)
                push!(target.displayed_names, path)
                push!(target.descriptions, "")
                push!(target.value_types, "NDArray")
            else
                collect_properties!(props, path, value, target)
            end
        end
    end
end

# Store or update a VariableStore for a given variable/subvariable.
function store_variable_data!(client, variable::VariableData)
    data = variable.data
    name = variable.name

    # `nothing` means the engine skipped this train (missing input, failed
    # body, or a variable that opted out by returning nothing). Skip silently.
    if isnothing(data)
        return
    end

    compression_ratio = NaN
    received_bytes = data isa AbstractArray ? sizeof(data) : 0
    if data isa CompressedArray
        # Don't decompress here: the payload is queued onto `store.updates` and
        # draw_plots decompresses only the most recent one, so frames that pile
        # up between renders are dropped before the expensive ZFP decode.
        received_bytes = sizeof(data.data) + (isnothing(data.nonfinite_mask) ? 0 : sizeof(data.nonfinite_mask))
        decompressed_bytes = prod(data.shape) * sizeof(data.original_eltype)
        compression_ratio = decompressed_bytes / received_bytes
    end

    # Unsubscribed array variables arrive as shape-only metadata. We store the
    # ArrayMetadata directly so plot buttons and type labels can read the
    # shape; real data only arrives after the user subscribes by opening a
    # plot. Skip data-flow updates in this path.
    if data isa ArrayMetadata
        if !haskey(client.variable_data, name)
            client.variable_data[name] = VariableStore(; data)
        end
        store = client.variable_data[name]
        store.data = data
        store.type = length(data.size) == 1 ? VariableType_Vector : VariableType_Array
        store.update_rate = variable.update_rate
        store.compression_ratio = compression_ratio
        store.received_bytes = received_bytes
        store.compress = variable.compress
        store.plot_specs = variable.plot_specs
        return
    end

    if !haskey(client.variable_data, name)
        if data isa Number
            values = CircularBuffer{Float64}(SCALAR_BUFFER_CAPACITY)
            tids = CircularBuffer{Int}(SCALAR_BUFFER_CAPACITY)
            client.variable_data[name] = VariableStore(; data=values, scalar_tids=tids)
        elseif data isa AbstractArray
            client.variable_data[name] = VariableStore(; data)
        elseif data isa CompressedArray
            # Seed with a buffer of the right eltype/shape so draw_plots can
            # decompress into it in place rather than allocating per train.
            client.variable_data[name] = VariableStore(; data=allocate_array(data))
        else
            @error "Unsupported variable type: $(typeof(data))"
            return
        end
    end

    store = client.variable_data[name]
    store.title = if !isnothing(variable.title)
        variable.title
    elseif data isa DimArray
        DD.label(data)
    elseif data isa CompressedArray && !isnothing(data.dims)
        data.dims.name
    else
        name
    end
    store.x_axis = variable.x_axis
    store.y_axis = variable.y_axis
    store.unit = variable.unit
    store.bin_resolution = variable.bin_resolution
    store.fixed_aspect = variable.fixed_aspect
    store.plot_type = variable.plot_type
    store.plot_specs = variable.plot_specs

    # Use explicit labels if provided, otherwise derive from DimArray or data type.
    # For 2D DimArrays the heatmap maps dim 1 → Y (rows) and dim 2 → X (cols).
    store.xlabel = if !isnothing(variable.xlabel)
        variable.xlabel
    elseif data isa DimMatrix
        DD.label(DD.dims(data)[2])
    elseif data isa DimArray
        DD.label(DD.dims(data)[1])
    elseif data isa CompressedArray && !isnothing(data.dims)
        string(length(data.shape) == 2 ? data.dims.dim_names[2] : data.dims.dim_names[1])
    elseif data isa Number
        "trainId"
    else
        ""
    end
    store.ylabel = if !isnothing(variable.ylabel)
        variable.ylabel
    elseif data isa DimMatrix
        DD.label(DD.dims(data)[1])
    elseif data isa DimArray
        DD.label(data)
    elseif data isa CompressedArray && !isnothing(data.dims)
        length(data.shape) == 2 ? string(data.dims.dim_names[1]) : data.dims.name
    else
        ""
    end

    type = if data isa Number
        VariableType_Scalar
    elseif data isa AbstractVector
        VariableType_Vector
    elseif data isa AbstractArray
        VariableType_Array
    elseif data isa CompressedArray
        length(data.shape) == 1 ? VariableType_Vector : VariableType_Array
    else
        VariableType_Unknown
    end
    push!(store.updates, (variable.tid, data, type))
    store.update_rate = variable.update_rate
    store.compression_ratio = compression_ratio
    store.received_bytes = received_bytes
    store.compress = variable.compress
end

@enum ParameterOwnerKind ParameterOwner_Group ParameterOwner_Postprocessor ParameterOwner_Global

struct ParameterOwner
    kind::ParameterOwnerKind
    var_name::String
    pp_name::Maybe{String}
    field_name::String
end

# Locate the node that owns a fully-qualified parameter in the loaded context.
# Returns nothing if not found, otherwise a ParameterOwner describing where
# the parameter lives.
function find_parameter_owner(client, param_name::String)
    for (var_name, var_data) in client.context.context_state
        if var_data["type"] === :group
            for (field_name, stored) in var_data["parameters"]
                if stored.name == param_name
                    return ParameterOwner(ParameterOwner_Group, var_name, nothing, field_name)
                end
            end
        elseif var_data["type"] === :variable
            for pp in var_data["postprocessors"]
                for (field_name, stored) in pp.params
                    if stored.name == param_name
                        return ParameterOwner(ParameterOwner_Postprocessor,
                                              var_name, pp.name, field_name)
                    end
                end
            end
        end
    end
    # Top-level `name = Parameter(...)` — present in context.parameters but not
    # nested under any group or postprocessor.
    if haskey(client.context.parameters, param_name) && !occursin('.', param_name)
        return ParameterOwner(ParameterOwner_Global, param_name, nothing, param_name)
    end
    return nothing
end

# Flatten the sources of all the inputs into a single sorted list for
# autocompletion, marking the sources whose name is served by more than one
# topic as ambiguous so that they're completed topic-qualified.
function build_source_list(sources)
    topics = Dict{String, Set{String}}()
    for source in sources
        push!(get!(Set{String}, topics, source.name), source.topic)
    end

    source_list = [SourceInfo(source.topic, source.name, source.class_id,
                              length(topics[source.name]) > 1)
                   for source in unique(sources)]
    return sort!(source_list; by=source -> source.name)
end

function handle_msg(state, msg, replied_to::Union{PendingRequest, Nothing}=nothing)
    client = state.client

    if msg isa Pong
        nothing

    elseif msg isa Stopped
        client.context.pipeline_status = PipelineStatus_Stopped

    elseif msg isa InputSources
        if msg.input_sources isa ExceptionMessage
            @error "Error from server with INPUTSOURCES" exception=msg.input_sources.text
            log_engine_error(state, "Failed to get the input sources", msg.input_sources.text)
        else
            client.source_list = build_source_list(Iterators.flatten(values(msg.input_sources)))

            sources_by_topic = Dict{String, Vector{SourceInfo}}()
            for source in client.source_list
                if !haskey(sources_by_topic, source.topic)
                    sources_by_topic[source.topic] = SourceInfo[]
                end
                push!(sources_by_topic[source.topic], source)
            end
            client.sources_by_topic = sources_by_topic
        end

    elseif msg isa EngineDir
        client.remote_engine_dir = msg.path

    elseif msg isa AvailableVariables
        if msg.variables isa ExceptionMessage
            @error "Error from server with AvailableVariables" exception=msg.variables.text
            log_engine_error(state, "Failed to get the available variables", msg.variables.text)
        else
            client.available_variables = msg.variables
        end

    elseif msg isa AvailableTrainmatchers
        trainmatchers = Dict{String, Vector{String}}()
        whitelisted = Set{KaraboDevice}()
        for (topic, ms) in msg.topic_trainmatchers
            names = String[]
            for (name, in_whitelist) in ms
                push!(names, name)
                if in_whitelist
                    push!(whitelisted, KaraboDevice(topic, name))
                end
            end
            trainmatchers[topic] = names
        end
        client.trainmatchers = trainmatchers
        client.trainmatcher_sources = build_source_list(
            [SourceInfo(topic, name, "TrainMatcher")
             for (topic, names) in trainmatchers for name in names])
        client.whitelisted_trainmatchers = whitelisted
        client.trainmatchers_request_status = RequestStatus_Idle

    elseif msg isa RoutingRules
        client.routing_rules = msg.rules
        client.routing_rules_request_status = RequestStatus_Idle

    elseif msg isa RemapRules
        client.remap_rules = msg.rules

    elseif msg isa DeviceSchema
        client.source_properties[(msg.topic, msg.name)] = schema_property_names(msg.schema)
        delete!(client.device_schema_requests, (msg.topic, msg.name))

    elseif msg isa DeviceProperty
        nothing

    elseif msg isa ContextInfo
        if msg.info isa Dict
            client.context.context_state = build_context_state(state, msg.info)
            client.context.source = msg.source
            client.context_path = msg.info["path"]
            filter!(kv -> haskey(client.context.context_state, kv.first), client.variable_data)
            # The new context has its own inputs, so the sources we complete
            # against have changed.
            get_input_sources(client)
            # A newly loaded context may pull in packages that register more
            # variables, so refresh the add-variable palette.
            get_available_variables(client)
        else
            @error "Context failed to load"
            log_engine_error(state, "Context failed to load", msg.info.text)
        end

        client.context.pipeline_status = msg.is_running ? PipelineStatus_Started : PipelineStatus_Stopped
    elseif msg isa TrainData
        for variable in msg.variables
            store_variable_data!(client, variable)

            for subvar in values(variable.subvariables)
                store_variable_data!(client, subvar)
            end
        end
    elseif msg isa ParameterChanged
        param = msg.parameter
        # Top-level dict covers globals (which aren't attached to any node).
        # Group/postprocessor params share instances with the node dicts, so
        # the walk below still updates them.
        if haskey(client.context.parameters, param.name)
            client.context.parameters[param.name].value = param.value
        end
        # The same parameter may appear under a node's "parameters" dict (group
        # params) or inside a postprocessor's `params`. Walk every node and
        # update wherever the full name matches.
        for var_data in values(client.context.context_state)
            params = get(var_data, "parameters", nothing)
            if params isa Dict
                for stored in values(params)
                    if stored isa Parameter && stored.name == param.name
                        stored.value = param.value
                    end
                end
            end
            for pp in get(var_data, "postprocessors", ())
                for stored in values(pp.params)
                    if stored isa Parameter && stored.name == param.name
                        stored.value = param.value
                    end
                end
            end
        end

        # An input's trainmatcher moved, so it's serving a different set of
        # sources now.
        if param.value isa KaraboDevice
            get_input_sources(client)
        end

        if client.pending_parameter_change == param.name
            client.pending_parameter_change = nothing
        end

        if client.pending_source_edit == param.name
            owner = find_parameter_owner(client, param.name)
            new_value = isnothing(owner) ? nothing : format_param_value(param.value)
            if !isnothing(new_value)
                if owner.kind == ParameterOwner_Group
                    set_group_param(state, owner.var_name, owner.field_name, new_value; reload=false)
                elseif owner.kind == ParameterOwner_Global
                    set_parameter_value(state, owner.var_name, new_value; reload=false)
                end
            end
            client.pending_source_edit = nothing
        end
    elseif msg isa PipelineStats
        client.context.channel_stats = msg.channel_stats
        client.context.input_rates = msg.input_rates
    elseif msg isa RemoteReplState
        client.remoterepl_mode[] = msg.enabled
        client.remoterepl_status = msg.enabled ? RemoteReplStatus_Running : RemoteReplStatus_Stopped
    elseif msg isa Ack
        if !isnothing(msg.error)
            @error "Server reported an error" exception=msg.error.text
            log_engine_error(state, "Server reported an error", msg.error.text)
        end

        if !isnothing(replied_to) && replied_to.msg_type == Start
            if isnothing(msg.error)
                client.context.pipeline_status = PipelineStatus_Started
            else
                client.context.pipeline_status = PipelineStatus_Stopped
            end
        end

        # On success, pending_parameter_change is cleared when the engine echoes
        # back a ParameterChanged. On failure, no echo will arrive so clear here.
        if !isnothing(replied_to) && replied_to.msg_type == ChangeParameter && !isnothing(msg.error)
            client.pending_parameter_change = nothing
            client.pending_source_edit = nothing
        end
    else
        @warn "Received unsupported message of type '$(typeof(msg))'"
    end
end

function handle_server(state)
    client = state.client
    port = client.embedded_engine ? client.engine.websocket_port : client.ws_forwarder.localport

    # If an SSH tunnel is used it can take a couple of seconds to set up, so we
    # allow multiple attempts.
    max_attempts = 10
    attempts = 0
    while attempts < max_attempts
        try
            # Note that we only support websockets available on localhost, either
            # because the server is running locally or because it's running remotely
            # and we've forwarded the port. Connecting to open servers is not
            # support for the moment.
            WebSockets.open("ws://localhost:$(port)"; suppress_close_error=true, maxframesize=Protocol.MAX_FRAME_SIZE) do ws
                client.websocket = ws

                # The first message we receive is our client ID
                id = WebSockets.receive(ws)
                client.client_id = id

                client.status = RemoteStatus_Connected
                send(client, GetEngineDir())
                get_input_sources(client)
                get_available_variables(client)
                get_trainmatchers(client)
                get_routing_rules(client)
                send(client, GetRemapRules())

                for msg_bytes in ws
                    buffer = IOBuffer(msg_bytes)
                    envelope::Envelope = deserialize(buffer)

                    replied_to = if !isnothing(envelope.reply_to)
                        pop!(client.pending_requests, envelope.reply_to, nothing)
                    else
                        nothing
                    end

                    callback = if !isnothing(envelope.reply_to)
                        pop!(client.engine_request_callbacks, envelope.reply_to, nothing)
                    else
                        nothing
                    end

                    try
                        @invokelatest handle_msg(state, envelope.msg, replied_to)
                        if !isnothing(callback)
                            @invokelatest callback(envelope.msg)
                        end
                    catch ex
                        @error "Error handling message!" exception=(ex, catch_backtrace())
                    end
                end

                @info "Connection to server closed ❌"
            end

            # Break from the attempt-retry loop after the connection has been
            # closed normally.
            break
        catch ex
            # If the client was already connected then we don't try again
            if !isnothing(client.websocket)
                @error "Client websocket loop exited with exception" exception=(ex, catch_backtrace())
                break
            end

            # If the port is not bound yet (e.g. if the SSH tunnel hasn't been
            # set up), we allow it to fail and retry.
            if ex isa HTTP.ConnectError
                attempts += 1
                @warn "Connection to server attempt $(attempts) failed..."
                sleep(2)
            else
                client.last_error = Util.exception2str(ex, catch_backtrace())
                client.status = RemoteStatus_Error
            end
        end
    end

    # If we've reached the maximum possible number of attempts, error out
    if client.status != RemoteStatus_Connected && attempts == max_attempts
        client.last_error = "Connection to server failed after $(attempts) attempts."
        client.status = RemoteStatus_Error

        # Call the shutdown function to ensure that the tunnel is killed too
        disconnect_engine(state, false)
    elseif client.status != RemoteStatus_Disconnecting
        # Otherwise we've disconnected normally. Don't overwrite
        # RemoteStatus_Disconnecting since disconnect_engine() manages
        # that transition.
        client.status = RemoteStatus_Unconnected
    end
end

function get_input_sources(client)
    send(client, GetInputSources())
end

function get_available_variables(client)
    send(client, GetVariables())
end

function get_trainmatchers(client)
    send(client, GetTrainmatchers())
    client.trainmatchers_request_status = RequestStatus_Waiting
end

function get_routing_rules(client)
    send(client, GetRoutingRules())
    client.routing_rules_request_status = RequestStatus_Waiting
end

# Returns (topic, device, classId) for the device referenced by `source`, or
# ("", device, "") if the device isn't reported by any input.
function source_device_info(client::ClientState, source::String)
    sep = find_separator(source)
    head = isnothing(sep) ? source : source[1:sep-1]
    topic_hint, device = split_topic(head)

    idx = findfirst(client.source_list) do known
        known.name == device && (isempty(topic_hint) || known.topic == topic_hint)
    end
    if isnothing(idx)
        return topic_hint, device, ""
    else
        known = client.source_list[idx]
        return known.topic, device, known.class_id
    end
end

source_device_class(client::ClientState, source::String) = source_device_info(client, source)[3]

# Apply a single rule to `source`. Returns the rewritten string, or nothing
# if the rule's source/device_class regexes don't both match.
# Returns (rewritten, pending). `rewritten` is the new source if the rule
# matched and produced a result, else nothing. `pending` is a request ID if
# the rule needs an in-flight device-property lookup to resolve.
function apply_remap_rule(client::ClientState, rule::RemapRule, source,
                          topic, device, device_class, property_ref::Ref{Any})
    if !occursin(Regex(rule.device_class), device_class)
        return nothing, nothing
    end

    pattern = Regex(rule.source)
    if !occursin(pattern, source)
        return nothing, nothing
    end

    return apply_remap_rule(Val(rule.kind), client, rule, source, topic, device, pattern, property_ref)
end

apply_remap_rule(::Val{RemapKind_Simple}, client, rule, source, topic, device, pattern, property_ref) =
    replace(source, pattern => SubstitutionString(rule.replacement)), nothing

# Rewrite `dev:output[prop]` to `<fast>[prop]@dev:output`, where `<fast>` is
# the sole entry of the device's `fastSources` property. Returns a pending
# request ID instead of a rewrite while the lookup is in flight; the response
# callback writes the value into `property_ref`.
function apply_remap_rule(::Val{RemapKind_Proxy}, client::ClientState, rule, source,
                          topic, device, pattern, property_ref::Ref{Any})
    if isnothing(property_ref[])
        id = send_with_callback(
            client, GetDeviceProperty(topic, device, "fastSources"),
            msg -> property_ref[] = msg.value)
        return nothing, id
    end

    value = property_ref[]
    if value isa Exception || !(value isa AbstractVector) || isempty(value)
        return nothing, nothing
    end

    bracket = findfirst('[', source)
    if isnothing(bracket)
        return nothing, nothing
    end

    proxied_source = string(value[1], source[bracket:end], "@", source[1:bracket-1])

    return proxied_source, nothing
end

# Walks the remap rules in order, applying every rule that matches
# cumulatively. Returns (source, pending); if pending is non-nothing the
# caller should wait for that request before re-running the remap. Proxy
# rules deposit the looked-up property value into `property_ref`.
function remap_source(client::ClientState, source::String, property_ref::Ref{Any})
    topic, device, device_class = source_device_info(client, source)
    for rule in client.remap_rules
        result, pending = apply_remap_rule(client, rule, source, topic, device, device_class,
                                           property_ref)
        if !isnothing(pending)
            return source, pending
        end
        if !isnothing(result)
            source = result
        end
    end
    return source, nothing
end

# Whether `property` is usable with `source`, given the device's schema property
# `names`. A property is valid if it's in the schema, or if it's what a schema
# property becomes once the remap rules run: committed sources store the
# remapped form (a camera's `data.image.pixels` is saved as `data.image.data`),
# which never appears in the schema itself.
function karabo_property_valid(client::ClientState, source::AbstractString, property::AbstractString, names)
    if property in names
        return true
    end

    topic, device, device_class = source_device_info(client, source)
    rules = filter(rule -> rule.kind == RemapKind_Simple, client.remap_rules)
    target = karabo_dep_string(nothing, source, property)

    for name in names
        remapped = karabo_dep_string(nothing, source, name)
        for rule in rules
            result, _ = apply_remap_rule(client, rule, remapped, topic, device, device_class,
                                         Ref{Any}(nothing))
            if !isnothing(result)
                remapped = result
            end
        end
        if remapped == target
            return true
        end
    end

    return false
end

function load_context(state)
    client = state.client
    send(client, LoadContext(client.context_path))
    client.context.pipeline_status = PipelineStatus_LoadingContext
end

function revise_engine(state)
    client = state.client
    if client.status == RemoteStatus_Connected && !isnothing(client.websocket)
        send(client, ReviseCode())
    end
end

function change_parameter(param::Parameter)
    client = state[].client
    send(client, ChangeParameter(param))
    client.pending_parameter_change = param.name
end

function start(state)
    for store in values(state.client.variable_data)
        store.update_rate = 0
    end

    send(state.client, Start())
    state.client.context.pipeline_status = PipelineStatus_Starting
end

function stop(state)
    send(state.client, Stop())
    state.client.context.pipeline_status = PipelineStatus_Stopping
end

function set_routing_rules(client, rules::AbstractVector{RoutingRule})
    client.routing_rules_set_request = send(client, SetRoutingRules(collect(rules)))
end

function set_debug_mode(state)
    client = state.client
    client.debug_mode_request = send(client, SetDebugMode(client.debug_mode[]))
end

function set_remoterepl(state)
    client = state.client
    client.remoterepl_status = RemoteReplStatus_Changing
    send(client, SetRemoteRepl(client.remoterepl_mode[]))
end

function send_subscriptions(client)
    variables = Dict{String, Float64}(name => sub.k
                                      for (name, sub) in client.subscriptions
                                      if sub.active)
    send(client, SetVariableSubscriptions(variables))
end

# Update the requested zfp accuracy `k` for a subscribed variable. Callers only
# invoke this on a real change (e.g. InputFloat edit), so we always send.
function set_subscription_k(state, name, k::Float64)
    client = state.client
    if isempty(name) || !haskey(client.subscriptions, name)
        return
    end
    client.subscriptions[name].k = k
    send_subscriptions(client)
end

function subscribe_variable(state, name; k::Maybe{Float64}=nothing)
    if isempty(name)
        return
    end

    client = state.client
    needs_send = false

    if !haskey(client.subscriptions, name)
        client.subscriptions[name] = SubscriptionState(; count=1,
                                                       k=something(k, -1))
        needs_send = true
    else
        sub = client.subscriptions[name]
        sub.count += 1
        if !sub.active
            sub.active = true
            needs_send = true
        end
        if !isnothing(k) && sub.k != k
            sub.k = k
            needs_send = true
        end
    end

    if needs_send
        send_subscriptions(client)
    end
end

# Empty/unknown names are no-ops so callers can pass uninitialised slots safely.
function unsubscribe_variable(state, name)
    if isempty(name) || !haskey(state.client.subscriptions, name)
        return
    end

    sub = state.client.subscriptions[name]
    sub.count -= 1
    if sub.count == 0
        sub.active = false
        send_subscriptions(state.client)
    end
end
