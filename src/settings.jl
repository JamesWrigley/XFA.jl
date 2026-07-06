function settings_path()
    config_dir = get(ENV, "XFA_CONFIG_DIR", joinpath(homedir(), ".xfa"))
    joinpath(config_dir, "settings.toml")
end

function load_settings()
    path = settings_path()
    isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
end

function write_settings(settings)
    path = settings_path()
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, settings; sorted=true)
    end
end

function save_section(section::String, data::Dict)
    settings = load_settings()
    settings[section] = data

    write_settings(settings)
end

function save_settings(state::GuiState, updated_field=nothing)
    fields_to_save = (:address, :engine_environment, :engine_working_dir, :client_type_current_item)
    if !isnothing(updated_field) && updated_field ∉ fields_to_save
        return
    end

    save_section("GuiState", Dict(
        "address" => state.address,
        "engine_environment" => state.engine_environment,
        "engine_working_dir" => state.engine_working_dir,
        "client_type" => state.client_type_current_item,
    ))
end

function save_settings(client::ClientState, updated_field=nothing)
    fields_to_save = (:context_path, :plot_counter)
    if isempty(client.context_path) || (!isnothing(updated_field) && updated_field ∉ fields_to_save)
        return
    end

    settings = load_settings()
    client_settings = get!(settings, "ClientState", Dict{String, Any}())

    # When only the context_path changed, save it without overwriting the
    # per-context data (node positions, plots, etc.) for the new path.
    if updated_field == :context_path
        client_settings["context_path"] = client.context_path

        write_settings(settings)
        return
    end

    plots = map(client.plots) do plot
        # Only single-layer Variable/Correlation plots have a stable round-trip
        # format today; richer multi-layer plots stay in-memory only.
        if length(plot.layers) == 1 && plot.layers[1] isa VariableLayer
            Dict("type" => "Plot", "name" => plot.layers[1].name,
                 "id" => plot.id, "dock_id" => plot.dock_id)
        elseif length(plot.layers) == 1 && plot.layers[1] isa CorrelationLayer
            Dict("type" => "CorrelationPlot", "id" => plot.id, "dock_id" => plot.dock_id)
        else
            nothing
        end
    end
    filter!(!isnothing, plots)

    ini_data = unsafe_string(ig.SaveIniSettingsToMemory(C_NULL))

    # Read existing contexts, update only this context's entry
    client_settings["context_path"] = client.context_path
    contexts = get!(client_settings, "contexts", Dict{String, Any}())

    contexts[client.context_path] = Dict(
        "plots" => plots,
        "plot_counter" => client.plot_counter,
        "saved_layout" => ini_data,
        "node_editor_state" => client.ne_settings,
    )

    write_settings(settings)
end

# The node editor's serialized state (positions, pan, zoom) for a context.
function load_node_editor_state(context_path)
    settings = load_settings()
    contexts = get(get(settings, "ClientState", Dict()), "contexts", Dict())
    ctx = get(contexts, context_path, Dict())
    return get(ctx, "node_editor_state", "")
end

# Persist just the editor state for a context without touching its other
# entries; used when switching away from a context before recreating the editor.
function save_node_editor_state(context_path, blob)
    if isempty(context_path)
        return
    end

    settings = load_settings()
    client_settings = get!(settings, "ClientState", Dict{String, Any}())
    contexts = get!(client_settings, "contexts", Dict{String, Any}())
    ctx = get!(contexts, context_path, Dict{String, Any}())
    ctx["node_editor_state"] = blob

    write_settings(settings)
end
