@kwdef mutable struct KaraboBridgeGuiState
    zmq_outputs::Union{Vector{String}, Exception, Nothing} = nothing
    zmq_outputs_request::Maybe{Int} = nothing
    selected_output::Cint = 0
    auto_select_output::Bool = false
end

function draw_variable_content(::Val{Symbol("XfaEngine.Context.KaraboBridge")}, name, var_data, gui_state)
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

function draw_postprocessor_params(::Val{Symbol("XfaEngine.Context.Histogram1D")}, pp, min_node_width)
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
