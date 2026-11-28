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
    fields_to_save = (:address, :julia_module, :engine_environment, :engine_working_dir, :client_type_current_item,
                      :font_percent)
    if !isnothing(updated_field) && updated_field ∉ fields_to_save
        return
    end

    save_section("GuiState", Dict(
        "address" => state.address,
        "julia_module" => state.julia_module,
        "engine_environment" => state.engine_environment,
        "engine_working_dir" => state.engine_working_dir,
        "client_type" => state.client_type_current_item,
        "font_percent" => state.font_percent,
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

    if isempty(client.layout_path)
        return
    end

    plots = [merge(plot_entry(plot.view.source), Dict("counter" => plot.counter)) for plot in client.plots]
    ini_data = unsafe_string(ig.SaveIniSettingsToMemory(C_NULL))

    @lock SETTINGS_LOCK begin
        # Read existing contexts, update only this context's entry
        settings = load_settings()
        client_settings = get!(settings, "ClientState", Dict{String, Any}())
        client_settings["context_path"] = client.context_path
        contexts = get!(client_settings, "contexts", Dict{String, Any}())

        contexts[client.layout_path] = Dict(
            "plots" => plots,
            "plot_counter" => client.plot_counter,
            "saved_layout" => ini_data,
            "node_editor_state" => client.ne_settings,
            "node_widths" => client.ne_node_widths,
        )

        write_settings(settings)
    end
end

# The saved entry of a context, empty if it has none.
function load_context_settings(context_path)
    settings = load_settings()
    contexts = get(get(settings, "ClientState", Dict()), "contexts", Dict())
    return get(contexts, context_path, Dict{String, Any}())
end

# The settings entry that restore_plot rebuilds a plot from
plot_entry(source::DefaultSpec) = Dict("type" => "Default", "variable" => source.variable)
plot_entry(source::CorrelationSpec) = Dict("type" => "Correlation", "selected" => copy(source.selected))
plot_entry(source::AdvertisedSpec) = Dict("type" => "Advertised", "variable" => source.variable, "name" => source.name)
plot_entry(source::RecordSpec) = Dict("type" => "Record", "output" => source.output, "key" => source.key)

# Rebuild a plot from its plot_entry, or nothing for entries in an older format
function restore_plot(entry)
    type = get(entry, "type", "")
    if type == "Default"
        variable_plot(entry["variable"], entry["counter"])
    elseif type == "Correlation"
        plot = correlation_plot(entry["counter"])
        copyto!(plot.view.source.selected, entry["selected"])
        plot
    elseif type == "Advertised"
        spec_plot(entry["variable"], entry["name"], entry["counter"])
    elseif type == "Record"
        record_plot(entry["output"], entry["key"], entry["counter"])
    else
        nothing
    end
end
