@kwdef mutable struct KaraboBridgeGuiState
    zmq_outputs::Union{Vector{String}, Exception, Nothing} = nothing
    zmq_outputs_request::Maybe{Int} = nothing
    selected_output::Cint = 0
    auto_select_output::Bool = false
end

function draw_variable_content(::Val{Symbol("XfaEngine.KaraboInput")}, name, var_data, gui_state)
    var_data["draw_parameters"] = false
    client = state[].client
    params = var_data["parameters"]

    if isnothing(gui_state)
        gui_state = KaraboBridgeGuiState()
    end

    ig.Text("Parameters:")

    tm_param = params["trainmatcher"]
    tm_modified, new_tm = draw_parameter("trainmatcher", tm_param)
    if tm_modified
        tm_param[] = new_tm
        @guiasync set_group_param(state[], name, "trainmatcher",
                                  "KaraboDevice(\"$(new_tm.topic)\", \"$(new_tm.name)\")")

        gui_state.zmq_outputs = nothing
        gui_state.selected_output = 0
        gui_state.auto_select_output = true
    end

    draw_parameter("manual_configuration", params["manual_configuration"])

    if params["manual_configuration"].value
        draw_parameter("address", params["address"])
    else
        # Fetch zmqOutputs from the trainmatcher device
        tm = params["trainmatcher"].value
        if !isempty(tm.topic) && !isempty(tm.name) && !is_pending(client, gui_state.zmq_outputs_request)
            if isnothing(gui_state.zmq_outputs)
                gui_state.zmq_outputs_request = send_with_callback(
                    client, GetDeviceProperty(tm.topic, tm.name, "zmqOutputs"),
                    msg -> begin
                        if msg.value isa Exception
                            gui_state.zmq_outputs = msg.value
                        else
                            gui_state.zmq_outputs = String[out["address"] for out in msg.value]
                        end
                        gui_state.zmq_outputs_request = nothing
                    end
                )
            end
        end

        if is_pending(client, gui_state.zmq_outputs_request)
            Spinner("Fetching outputs...")
        elseif gui_state.zmq_outputs isa Exception
            ig.TextColored(ig.ImVec4(1, 0.4, 0.4, 1), sprint(showerror, gui_state.zmq_outputs))
        elseif gui_state.zmq_outputs isa Vector
            if !isempty(gui_state.zmq_outputs)
                # After a trainmatcher change, push the first output through as the new address
                if gui_state.auto_select_output
                    gui_state.auto_select_output = false
                    gui_state.selected_output = 0
                    new_address = gui_state.zmq_outputs[1]
                    address_param = params["address"]
                    if address_param.value != new_address
                        client.pending_source_edit = address_param.name
                        change_parameter(Parameter(address_param.name, new_address))
                    end
                end

                # Sync combo selection with the current address parameter
                current_address = params["address"].value
                if !isempty(current_address)
                    found = findfirst(==(current_address), gui_state.zmq_outputs)
                    if !isnothing(found)
                        gui_state.selected_output = Cint(found - 1)
                    end
                end

                ig.SetNextItemWidth(350)
                idx = Ref(gui_state.selected_output)
                if CopyableCombo("Output", gui_state.zmq_outputs, idx)
                    gui_state.selected_output = idx[]
                    new_address = gui_state.zmq_outputs[idx[] + 1]
                    address_param = params["address"]
                    client.pending_source_edit = address_param.name
                    change_parameter(Parameter(address_param.name, new_address))
                end
            else
                ig.Text("No ZMQ outputs found")
            end
        end

        if !is_pending(client, gui_state.zmq_outputs_request)
            ig.SameLine()
            if ig.Button("Refresh##zmq_outputs_$(name)")
                gui_state.zmq_outputs = nothing
            end
        end
    end

    return gui_state
end

@kwdef mutable struct AttosecondState
    fits::Vector{Vector{Float64}} = Vector{Float64}[]
    xs::Vector{Float64} = Float64[]
    spectrum_name::String = ""
    last_tid::Int = -1
end

# Overlay hook called from draw_plot for each entry in client.variable_gui_states.
# Specialize on the gui_state type to add ImPlot series on top of the plot
# identified by `plot_name`. Called between BeginPlot/EndPlot.
plot_overlay(::Any, plot_name) = nothing

function plot_overlay(s::AttosecondState, plot_name)
    if s.spectrum_name != plot_name
        return
    end
    for (i, curve) in enumerate(s.fits)
        ImPlot.PlotLine("spike $i fit", s.xs, curve)
    end
end

function draw_variable_overlays(plot_name)
    for (_, gui_state) in state[].client.variable_gui_states
        plot_overlay(gui_state, plot_name)
    end
end

# Resolve the variable name connected to a group's Parameter{Dependency} field.
function group_dep_variable(var_data, field::AbstractString)
    for (attr_id, (_, dep)) in var_data["dependencies"]
        if get(var_data["dep_field_names"], attr_id, nothing) == field
            return dep.name
        end
    end
    return nothing
end

# spike_fitting returns a (params, n_spikes) matrix where each column is
# (y0, A, mu, sigma) for one spike. Sample each gaussian over xs and store
# the resulting curve in state.fits.
function update_spike_fits!(gui_state::AttosecondState, params::AbstractMatrix, xs)
    resize!(gui_state.xs, length(xs))
    copyto!(gui_state.xs, xs)

    n_spikes = size(params, 2)
    resize!(gui_state.fits, n_spikes)
    for k in 1:n_spikes
        curve = isassigned(gui_state.fits, k) ? gui_state.fits[k] : Float64[]
        resize!(curve, length(xs))
        y0, A, μ, σ = params[1, k], params[2, k], params[3, k], params[4, k]
        for (i, x) in enumerate(xs)
            curve[i] = gaussian(x, y0, A, μ, σ)
        end
        gui_state.fits[k] = curve
    end
end

function draw_variable_content(::Val{Symbol("AttosecondFit")}, name, var_data, gui_state)
    client = state[].client
    if isnothing(gui_state)
        gui_state = AttosecondState()
        subscribe_variable(state[], "$(name).spike_fitting")
    end

    # Recompute fit curves whenever spike_fitting produces a new train.
    gui_state.spectrum_name = something(group_dep_variable(var_data, "spectrum"), "")
    fit_store = get(client.variable_data, "$(name).spike_fitting", nothing)
    if !isnothing(fit_store) && fit_store.data isa AbstractMatrix &&
       fit_store.trainId != gui_state.last_tid
        spectrum_store = isempty(gui_state.spectrum_name) ?
            nothing : get(client.variable_data, gui_state.spectrum_name, nothing)
        xs = if !isnothing(spectrum_store) && spectrum_store.data isa AbstractVector
            isnothing(spectrum_store.x_axis) ?
                (1:length(spectrum_store.data)) : spectrum_store.x_axis
        else
            1:size(fit_store.data, 2)
        end
        update_spike_fits!(gui_state, fit_store.data, xs)
        gui_state.last_tid = fit_store.trainId
    end

    if ig.Button("Plot fits##$(name)")
        spectrum_name = group_dep_variable(var_data, "spectrum")
        if !isnothing(spectrum_name)
            push!(client.plots, variable_plot(spectrum_name, client.plot_counter))
            client.plot_counter += 1
        end
    end
    return gui_state
end

function draw_postprocessor_params(::Val{Symbol("XfaContext.Histogram1D")}, pp, min_node_width)
    param_order = ("nbins", "binedges", "normalize", "windowed", "buffer_size")
    for param_name in param_order
        if param_name == "buffer_size" && !pp.params["windowed"].value
            continue
        end
        if haskey(pp.params, param_name)
            draw_parameter(param_name, pp.params[param_name]; min_node_width)
        end
    end
end
