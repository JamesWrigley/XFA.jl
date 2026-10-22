config_dir() = get(ENV, "XFA_CONFIG_DIR", joinpath(homedir(), ".xfa"))

settings_path() = joinpath(config_dir(), "settings.toml")

# Settings are saved from both the GUI thread and client tasks
const SETTINGS_LOCK = ReentrantLock()

function load_settings()
    path = settings_path()
    @lock SETTINGS_LOCK isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
end

function write_settings(settings)
    path = settings_path()
    mkpath(dirname(path))
    @lock SETTINGS_LOCK begin
        # Write to a temp file and rename it over the original so other GUI
        # processes never see a partially-written file. It's a bit of a hack, we
        # should figure out how to do this properly.
        tmp_path = tempname(dirname(path); cleanup=false)
        open(tmp_path, "w") do io
            TOML.print(io, settings; sorted=true)
        end
        mv(tmp_path, path; force=true)
    end
end

function save_section(section::String, data::Dict)
    @lock SETTINGS_LOCK begin
        settings = load_settings()
        settings[section] = data

        write_settings(settings)
    end
end

function save_settings(state::GuiState, updated_field=nothing)
    fields_to_save = (:address, :julia_module, :engine_environment, :engine_working_dir, :client_type_current_item)
    if !isnothing(updated_field) && updated_field ∉ fields_to_save
        return
    end

    save_section("GuiState", Dict(
        "address" => state.address,
        "julia_module" => state.julia_module,
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

    # When only the context_path changed, save it without overwriting the
    # per-context data (node positions, plots, etc.) for the new path.
    if updated_field == :context_path
        @lock SETTINGS_LOCK begin
            settings = load_settings()
            client_settings = get!(settings, "ClientState", Dict{String, Any}())
            client_settings["context_path"] = client.context_path

            write_settings(settings)
        end
        return
    end

    plots = map(client.plots) do plot
        # Only default and correlation plots have a stable round-trip format
        # today; advertised specs stay in-memory only.
        source = plot.view.source
        if source isa DefaultSpec
            Dict("type" => "Plot", "name" => source.variable,
                 "id" => plot.id, "dock_id" => plot.dock_id)
        elseif source isa CorrelationSpec
            Dict("type" => "CorrelationPlot", "id" => plot.id, "dock_id" => plot.dock_id)
        else
            nothing
        end
    end
    filter!(!isnothing, plots)

    ini_data = unsafe_string(ig.SaveIniSettingsToMemory(C_NULL))

    @lock SETTINGS_LOCK begin
        # Read existing contexts, update only this context's entry
        settings = load_settings()
        client_settings = get!(settings, "ClientState", Dict{String, Any}())
        client_settings["context_path"] = client.context_path
        contexts = get!(client_settings, "contexts", Dict{String, Any}())

        contexts[client.context_path] = Dict(
            "plots" => plots,
            "plot_counter" => client.plot_counter,
            "saved_layout" => ini_data,
            "node_editor_state" => client.ne_settings,
            "node_widths" => client.ne_node_widths,
        )

        write_settings(settings)
    end
end

# The node editor's serialized state (positions, pan, zoom) for a context.
function load_node_editor_state(context_path)
    settings = load_settings()
    contexts = get(get(settings, "ClientState", Dict()), "contexts", Dict())
    ctx = get(contexts, context_path, Dict())
    widths = Dict{String, Float32}(name => Float32(w) for (name, w) in get(ctx, "node_widths", Dict()))
    return get(ctx, "node_editor_state", ""), widths
end

# Persist just the editor state for a context without touching its other
# entries; used when switching away from a context before recreating the editor.
function save_node_editor_state(context_path, blob, widths)
    if isempty(context_path)
        return
    end

    @lock SETTINGS_LOCK begin
        settings = load_settings()
        client_settings = get!(settings, "ClientState", Dict{String, Any}())
        contexts = get!(client_settings, "contexts", Dict{String, Any}())
        ctx = get!(contexts, context_path, Dict{String, Any}())
        ctx["node_editor_state"] = blob
        ctx["node_widths"] = widths

        write_settings(settings)
    end
end
