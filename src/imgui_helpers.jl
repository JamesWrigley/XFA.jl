mutable struct SafeInputTextState
    buffer::Vector{UInt8}
    reference_text::String
end

function SafeInputTextState(max_len::Int, reference_text::String)
    buffer = zeros(UInt8, max_len + 1) # Add 1 for the null pointer
    SafeInputTextState(buffer, reference_text)
end

const safe_input_text_cache = Dict{UInt32, SafeInputTextState}()


# The last item's rect, mapped to screen space. Inside the node editor the item
# rect is in canvas-local space (the canvas remaps the ImGui coordinate system),
# so a popup positioned from it lands in the wrong place unless converted.
function editor_item_rect()
    item_min = ig.GetItemRectMin()
    item_max = ig.GetItemRectMax()
    if ne.GetCurrentEditor() != C_NULL
        item_min = ne.CanvasToScreen(item_min)
        item_max = ne.CanvasToScreen(item_max)
    end
    return item_min, item_max
end


function MenuItem(label::String, selected::Ref{Bool}, enabled::Bool=true)
    ig.MenuItem(label, C_NULL, selected, enabled)
end

# Based on: https://github.com/ocornut/imgui/issues/718#issuecomment-1249822993
function EditableComboBox(label, text, completions;
                          max_len=63, flags=ig.ImGuiInputTextFlags_None)
    flags |= ig.ImGuiInputTextFlags_EnterReturnsTrue

    # Initialize a buffer to hold the input, and copy the initial text into it
    input = zeros(UInt8, max_len + 1)
    Util.strcpy!(input, text)
    enter_pressed = ig.InputText(label, pointer(input), max_len, flags)

    if ig.IsItemActivated()
        ig.OpenPopup(label)
    end

    item_min, item_max = editor_item_rect()
    ig.SetNextWindowPos(ImVec2(item_min.x, item_max.y))
    popup_flags = ig.ImGuiWindowFlags_NoTitleBar
    popup_flags |= ig.ImGuiWindowFlags_NoMove
    popup_flags |= ig.ImGuiWindowFlags_NoResize

    edited = false
    current_input = unsafe_string(pointer(input))
    if ig.BeginPopup(label, popup_flags)
        # If the widget isn't active or we've finished editing, close the popup
        # Otherwise, build the list of completions
        for option in completions
            if ig.Selectable(option)
                ig.ClearActiveID()
                Util.strcpy!(input, option)
            end
        end

        if enter_pressed || (!ig.IsItemActive() && !ig.IsWindowFocused())
            ig.CloseCurrentPopup()
            edited = true
        end

        ig.EndPopup()
    end

    return edited, unsafe_string(pointer(input))
end

function SafeInputText(label; max_len=127, hint="", current_text="", password=false, reset=false,
                       callback=C_NULL, user_data=C_NULL, validator=nothing)
    id = ig.GetID(label)
    if !haskey(safe_input_text_cache, id) || reset
        safe_input_text_cache[id] = SafeInputTextState(max_len, current_text)
        Util.strcpy!(safe_input_text_cache[id].buffer, current_text)
    end

    state = safe_input_text_cache[id]

    if state.reference_text != current_text
        state.reference_text = current_text
        Util.strcpy!(state.buffer, current_text)
    end

    flags = ig.ImGuiInputTextFlags_EnterReturnsTrue
    if password
        flags |= ig.ImGuiInputTextFlags_Password
    end
    if callback !== C_NULL
        flags |= ig.ImGuiInputTextFlags_CallbackAlways
    end

    modified = unsafe_string(pointer(state.buffer)) != current_text
    validation_error = if !isnothing(validator) && modified
        validator(unsafe_string(pointer(state.buffer)))
    else
        nothing
    end

    if !isnothing(validation_error)
        ig.PushStyleColor(ig.ImGuiCol_FrameBg, ig.IM_COL32(180, 40, 40, 255))
    elseif modified
        ig.PushStyleColor(ig.ImGuiCol_FrameBg, ig.IM_COL32(143, 98, 0, 255))
    end
    ret = ig.InputTextWithHint(label, hint, pointer(state.buffer), length(state.buffer),
                               flags, callback, user_data)
    if !isnothing(validation_error)
        ig.PopStyleColor()
        ig.Text(validation_error)
        ret = false
    elseif modified
        ig.PopStyleColor()
    end

    return ret, unsafe_string(pointer(state.buffer))
end

function BorderedText(text; color=IM_COL32(255, 0, 0, 255), thickness=2.0, padding=ImVec2(4, 4))
    draw_list = ig.GetWindowDrawList()
    cursor = ig.GetCursorScreenPos()
    text_size = ig.CalcTextSize(text)
    p_min = ImVec2(cursor.x - padding.x, cursor.y - padding.y)
    p_max = ImVec2(cursor.x + text_size.x + padding.x, cursor.y + text_size.y + padding.y)
    ig.AddRect(draw_list, p_min, p_max, color, 0.0, thickness)
    ig.Text(text)
end

function BoxedText(label, text)
    if ig.BeginChild(label, ImVec2(0, 0), true,
                     ig.ImGuiWindowFlags_HorizontalScrollbar)
        button_w, _ = RowCopyButton_size()
        win_pos = ig.GetWindowPos()
        avail_w = ig.GetContentRegionAvail().x
        ig.SetCursorScreenPos(ImVec2(win_pos.x + avail_w - button_w - 2, win_pos.y + 6))
        CopyButton("$label-copy", text)
        ig.SetCursorPos(ImVec2(0, 0))
        ig.TextUnformatted(text)
        ig.EndChild()
    end
end

# Diabolically stolen from: https://github.com/ocornut/imgui/issues/1901#issuecomment-400563921
function Spinner(text="")
    characters = "|/-\\"
    idx = 1 + (trunc(Int, time() / 0.07) & (length(characters) - 1))

    ig.PushStyleColor(ig.ImGuiCol_Text, IM_COL32(255, 255, 255, 150))
    ig.Text(text * " " * characters[idx])
    ig.PopStyleColor()
end

macro guiasync(expr)
    return :(errormonitor(Threads.@spawn $(esc(expr))))
end

macro Disabled(cond, expr)
    return quote
        local disable = $(esc(cond))

        if disable
            ig.BeginDisabled(disable)
        end

        $(esc(expr))

        if disable
            ig.EndDisabled()
        end
    end
end

# Stolen from: https://github.com/ocornut/imgui/pull/4675
function IsItemDisabled()
    imgui_ctx = unsafe_load(ig.GetCurrentContext())
    return (imgui_ctx.LastItemData.ItemFlags & ig.ImGuiItemFlags_Disabled) != 0
end

const elided_text_states = Dict{UInt32, ElidedTextState}()

"""
    fuzzy_match(query, text) -> (matches::Bool, score::Int)

Simple fuzzy match: checks if all characters in `query` appear in `text` in
order (case-insensitive). Score is based on consecutive matches and early
positions.
"""
function fuzzy_match(query::AbstractString, text::AbstractString)
    q = filter(!isspace, lowercase(query))
    t = lowercase(text)

    # Substring match: rank these above any fuzzy result, by match position
    # first and then text length. The 1_000_000 floor leaves plenty of room
    # below the position penalty without colliding with fuzzy scores.
    substr = findfirst(q, t)
    if !isnothing(substr)
        return true, 1_000_000 - 100 * first(substr) - length(t)
    end

    qi = 1
    score = 0
    prev_match_pos = 0

    for (ti, tc) in enumerate(t)
        qi > length(q) && break
        if tc == q[qi]
            # Bonus for consecutive matches and early positions
            score += (ti == prev_match_pos + 1) ? 10 : 1
            score += max(0, length(t) - ti)
            prev_match_pos = ti
            qi += 1
        end
    end

    return qi > length(q), score
end

"""
    fuzzy_match(query, completions, completion_text; n=20)

Fuzzy-match `query` against `completions`, returning the top `n` results sorted
by score (descending). Uses a partial sort to avoid scoring more than necessary.
"""
function fuzzy_match(query::AbstractString, completions::AbstractVector, completion_text::Base.Callable; n=40)
    scored = Tuple{Int, Any}[]
    for item in completions
        matched, score = fuzzy_match(query, completion_text(item))
        matched || continue
        if length(scored) < n
            push!(scored, (score, item))
        elseif score > first(scored[1])
            scored[1] = (score, item)
            sort!(scored; by=first)  # keep min at front for cheap replacement
        end
    end

    # The partial-top-N strategy above leaves the buffer min-at-front (or in
    # insertion order if it never filled). Sort descending so callers can iterate
    # best-first.
    sort!(scored; by=first, rev=true)
    return scored
end

"""
    draw_autocomplete_popup(label, state, ac, input_min, input_max)
        -> (selected::Maybe{String}, hovered::Bool)

Draw the autocomplete popup under the input whose screen-space rect is
`input_min`/`input_max`. Returns the selected completion text if one was chosen,
`nothing` otherwise.

Inside the node canvas the caller must defer this to `draw_dag`'s Suspend/Resume
block via `client.completion_popup` (see `CompletionPopup`); outside it, the
widget calls it directly.
"""
function draw_autocomplete_popup(label, state::ElidedTextState, ac::CompletionResult,
                                 input_min::ImVec2, input_max::ImVec2)
    popup_label = "##autocomplete-$(label)"

    scored = if ac.query == state.cached_query && ac.source == state.cached_source && !isempty(state.cached_scored)
        state.cached_scored
    else
        state.cached_query = ac.query
        state.cached_source = ac.source
        state.cached_scored = fuzzy_match(ac.query, ac.items, identity)
    end

    result = nothing
    popup_hovered = false

    if !isempty(scored)
        row_height = ig.GetTextLineHeightWithSpacing()
        max_rows = 8
        popup_height = min(length(scored), max_rows) * row_height + 2 * unsafe_load(ig.GetStyle().WindowPadding.y)
        popup_width = 600

        ig.SetNextWindowPos(ImVec2(input_min.x, input_max.y))
        ig.SetNextWindowSize(ImVec2(popup_width, popup_height))

        flags = ig.ImGuiWindowFlags_NoTitleBar | ig.ImGuiWindowFlags_NoResize |
                ig.ImGuiWindowFlags_NoMove | ig.ImGuiWindowFlags_NoFocusOnAppearing |
                ig.ImGuiWindowFlags_NoSavedSettings | ig.ImGuiWindowFlags_Tooltip

        if ig.Begin(popup_label, C_NULL, flags)
            # The editor window is focused, so it sits ahead of this never-focused
            # popup in the hit-test order and would steal hover where they overlap.
            # Bring the popup to the front of that order (without taking keyboard
            # focus) so its rows hover/click normally.
            ig.BringWindowToDisplayFront(ig.GetCurrentWindow())
            popup_hovered = ig.IsWindowHovered() || ig.IsWindowFocused()

            # Keyboard navigation
            if ig.IsKeyPressed(ig.ImGuiKey_DownArrow)
                state.selected_idx = min(state.selected_idx + 1, length(scored))
            elseif ig.IsKeyPressed(ig.ImGuiKey_UpArrow)
                state.selected_idx = max(state.selected_idx - 1, 1)
            end

            for (i, (_, item)) in enumerate(scored)
                is_selected = (i == state.selected_idx)
                ig.PushID(i)
                ig.SetNextItemAllowOverlap()
                if ac.renderer(item, i, is_selected)
                    result = item
                end
                RowCopyButton("ac-$i", item, popup_width)
                if is_selected && (ig.IsKeyPressed(ig.ImGuiKey_Tab) || ig.IsKeyPressed(ig.ImGuiKey_Enter))
                    result = item
                end
                ig.PopID()
            end
        end
        ig.End()
    end

    state.selected_idx = clamp(state.selected_idx, 1, max(length(scored), 1))

    return result, popup_hovered
end

function ElidedText(label::AbstractString, text::AbstractString;
                    max_chars::Int=30, editable::Bool=false,
                    focus::Bool=false,
                    completions=nothing,
                    callback=C_NULL, user_data=C_NULL,
                    validator=nothing)
    field_state = get!(ElidedTextState, elided_text_states, ig.GetID(label))
    if focus && field_state.edit == ElidedEditState_NoEdit
        field_state.edit = ElidedEditState_WantEdit
    end

    min_width = ig.CalcTextSize("m").x * 13  # minimum clickable width

    if editable && field_state.edit != ElidedEditState_NoEdit
        just_started = field_state.edit == ElidedEditState_WantEdit
        if just_started
            ig.SetKeyboardFocusHere()
            field_state.edit = ElidedEditState_Edit
            field_state.selected_idx = 1
        end
        ig.SetNextItemWidth(max(min_width, ig.CalcTextSize(text).x + 40))
        edited, new_text = SafeInputText("##elided-$(label)"; current_text=text, reset=just_started,
                                         callback, user_data, validator)
        lost_focus = !just_started && ig.IsItemDeactivated() && !ig.IsItemActive()
        input_min, input_max = editor_item_rect()

        # Draw autocomplete popup if completions are provided
        ac_result = nothing
        ac_hovered = false
        ac_showing = false
        if !isnothing(completions)
            ac = completions(new_text)

            if ne.GetCurrentEditor() == C_NULL
                ac_result, ac_hovered = draw_autocomplete_popup(label, field_state, ac,
                                                                input_min, input_max)
            else
                # Inside the node canvas the popup has to be drawn after EndNode
                # (see CompletionPopup): pick up the previous frame's outcome, then
                # record this frame's request for draw_dag to draw.
                popup = state[].client.completion_popup
                if popup.drawn_label == label
                    popup.drawn_label = nothing
                    ac_result = popup.result
                    ac_hovered = popup.hovered
                    popup.result = nothing
                end

                popup.label = label
                popup.state = field_state
                popup.completions = ac
                popup.input_min = input_min
                popup.input_max = input_max

                # The deferred popup handles Enter a frame late, so the input's own
                # Enter-commit is held while it has rows, else the raw text commits first.
                ac_showing = !isempty(field_state.cached_scored)
            end
        end

        if !isnothing(ac_result)
            field_state.edit = ElidedEditState_NoEdit
            return true, ac_result
        elseif edited && ac_showing
            # Enter went to the deferred popup, which also deactivates the input; keep
            # the edit alive so next frame picks up the popup's selection (mirrors how
            # ac_hovered holds it open for a row click).
        elseif edited
            field_state.edit = ElidedEditState_NoEdit
            if new_text != text && !isempty(new_text)
                return true, new_text
            end
        elseif ig.IsKeyPressed(ig.ImGuiKey_Escape)
            field_state.edit = ElidedEditState_NoEdit
        elseif lost_focus && !ac_hovered
            field_state.edit = ElidedEditState_NoEdit
        end
    else
        elide = length(text) > max_chars
        display_text = elide ? text[1:max_chars] * "…" : text

        if editable
            text_size = ig.CalcTextSize(display_text)
            padding = ImVec2(2, 2)
            cursor = ig.GetCursorPos()
            ig.SetCursorPos(ImVec2(cursor.x, cursor.y - padding.y))
            ig.InvisibleButton("##elided-btn-$(label)", ImVec2(max(text_size.x, min_width) + 2 * padding.x, text_size.y + 2 * padding.y))
            hovered = ig.IsItemHovered()
            clicked = ig.IsItemClicked()

            draw_list = ig.GetWindowDrawList()
            p_min = ig.GetItemRectMin()
            p_max = ig.GetItemRectMax()
            if hovered
                ig.AddRectFilled(draw_list, p_min, p_max, ig.IM_COL32(60, 60, 80, 255))
            end
            text_pos = ImVec2(p_min.x + padding.x, p_min.y + padding.y)
            ig.AddText(draw_list, text_pos, ig.GetColorU32(ig.ImGuiCol_Text), display_text)

            if hovered && elide
                node_tooltip(text)
            end
            if clicked
                field_state.edit = ElidedEditState_WantEdit
            end
        else
            ig.Text(display_text)
            if ig.IsItemHovered() && elide
                node_tooltip(text)
            end
        end
    end

    return false, text
end

ElidedText(text::AbstractString; max_chars::Int=30) = ElidedText("", text; max_chars)

function InfoMarker(message::AbstractString, marker::AbstractString="?")
    ig.TextDisabled("[$(marker)]")
    if ig.IsItemHovered()
        node_tooltip(message)
    end
end

function CopyButton(label, text)
    ig.PushStyleVar(ig.ImGuiStyleVar_FramePadding, ImVec2(1, 1))
    ig.PushStyleVar(ig.ImGuiStyleVar_FrameBorderSize, 0)
    ig.PushStyleColor(ig.ImGuiCol_Button, ImVec4(0, 0, 0, 0))
    ig.PushStyleColor(ig.ImGuiCol_ButtonActive, ImVec4(0, 0, 0, 0))
    if ig.Button("\uf0c5##$(label)")
        ig.SetClipboardText(text)
    end
    ig.PopStyleColor(2)
    ig.PopStyleVar(2)
end

function RowCopyButton_size()
    width = ig.CalcTextSize("\uf0c5").x + 2
    height = ig.GetFontSize() + 1

    width, height
end

# Right-aligned copy button that appears on hover for the current row. Call this
# after drawing the row content (Selectable + text). The Selectable must have
# been created with SetNextItemAllowOverlap().
function RowCopyButton(label, copy_text, popup_width)
    width, height = RowCopyButton_size()

    row_min = ig.GetItemRectMin()
    row_max = ig.GetItemRectMax()
    ig.SameLine(0, 0)
    ig.SetCursorScreenPos(ImVec2(row_min.x + popup_width - width, row_min.y))
    if ig.IsMouseHoveringRect(row_min, ImVec2(row_min.x + popup_width, row_max.y))
        CopyButton(label, copy_text)
    else
        ig.Dummy(ImVec2(width, height))
    end
end

# A dropdown where each item has a copy-to-clipboard button, and the preview shows
# a copy button on hover. Returns true if the selection changed.
#
# Drawn as a button + popup rather than BeginCombo: a combo dropdown can't open
# mid-node. Inside the node canvas the dropdown is deferred to after EndNode under
# Suspend/Resume (see draw_dag) via client.combo_popup; outside it's drawn here.
function CopyableCombo(label, items, selected_idx::Ref{Cint})
    client = state[].client
    changed = false

    # Pick up a selection produced by a deferred draw on a previous frame.
    popup = client.combo_popup
    if popup.result_label == label
        popup.result_label = nothing
        if popup.result_index != selected_idx[]
            selected_idx[] = popup.result_index
            changed = true
        end
    end

    sel = selected_idx[] + 1
    preview = 1 <= sel <= length(items) ? items[sel] : ""
    button_width, _ = RowCopyButton_size()
    popup_id = "combo-popup-$label"
    in_editor = ne.GetCurrentEditor() != C_NULL

    # Elide the preview to leave room for the dropdown arrow on the right.
    item_w = ig.CalcItemWidth()
    arrow_w = ig.GetFrameHeight()
    avail = item_w - arrow_w - 2 * unsafe_load(ig.GetStyle().FramePadding.x)
    display = preview
    if ig.CalcTextSize(display).x > avail
        while !isempty(display) && ig.CalcTextSize(display * "…").x > avail
            display = display[1:prevind(display, lastindex(display))]
        end
        display *= "…"
    end

    # No AllowOverlap: inside the node editor an overlap-allowing button yields to
    # the editor's node-interaction area and stops responding, so the copy
    # affordance is a manual hit-tested sub-region rather than a second button.
    clicked = ig.Button("$display##$label", ImVec2(item_w, 0))

    rect_min = ig.GetItemRectMin()
    rect_max = ig.GetItemRectMax()
    # IsItemHovered (not IsMouseHoveringRect) so the copy icon respects focus and
    # isn't shown when another window or popup is on top.
    hovered = ig.IsItemHovered()

    # Anchor the dropdown under the button. In the editor the rect is canvas-local,
    # so convert to screen space for the deferred (suspended) draw.
    anchor = ImVec2(rect_min.x, rect_max.y)
    if in_editor
        anchor = ne.CanvasToScreen(anchor)
    end

    # Copy sub-region, just left of the dropdown arrow.
    copy_min = ImVec2(rect_max.x - arrow_w - button_width - 4, rect_min.y)
    copy_max = ImVec2(rect_max.x - arrow_w - 4, rect_max.y)
    over_copy = hovered && ig.IsMouseHoveringRect(copy_min, copy_max)

    if clicked
        if over_copy
            ig.SetClipboardText(preview)
        elseif in_editor
            popup.label = label
            popup.items = items
            popup.anchor = anchor
            popup.width = item_w
            popup.trigger = true
        else
            ig.OpenPopup(popup_id)
        end
    end

    draw_list = ig.GetWindowDrawList()
    pad_y = unsafe_load(ig.GetStyle().FramePadding.y)
    # Dropdown arrow at the right edge, matching a real combo.
    ig.RenderArrow(draw_list, ImVec2(rect_max.x - arrow_w + pad_y, rect_min.y + pad_y),
                   ig.GetColorU32(ig.ImGuiCol_Text), ig.ImGuiDir_Down, 1f0)
    # Copy icon on hover, brighter when the cursor is over it.
    if hovered
        col = ig.GetColorU32(over_copy ? ig.ImGuiCol_Text : ig.ImGuiCol_TextDisabled)
        ig.AddText(draw_list, ImVec2(copy_min.x, rect_min.y + pad_y), col, "\uf0c5")
    end

    # Outside the editor the dropdown is drawn here; inside, draw_dag draws it.
    if !in_editor
        ig.SetNextWindowPos(anchor)
        new_idx = draw_combo_popup(popup_id, items, item_w)
        if !isnothing(new_idx) && new_idx != selected_idx[]
            selected_idx[] = new_idx
            changed = true
        end
    end

    return changed
end

# Draws the CopyableCombo dropdown list. Returns the chosen index, or nothing if
# no selection was made this frame. The caller owns OpenPopup and Suspend/Resume.
function draw_combo_popup(popup_id, items, min_width)
    result = nothing
    if ig.BeginPopup(popup_id)
        button_width, _ = RowCopyButton_size()
        text_w = maximum((ig.CalcTextSize(name).x for name in items); init=0f0)
        # Subtract the window padding so the popup's outer width matches the widget.
        inner_width = min_width - 2 * unsafe_load(ig.GetStyle().WindowPadding.x)
        popup_w = max(inner_width, text_w + button_width + 24)
        for (i, name) in enumerate(items)
            ig.SetNextItemAllowOverlap()
            if ig.Selectable("$name##$popup_id-$i", false, 0, ImVec2(popup_w, 0))
                result = Cint(i - 1)
                ig.CloseCurrentPopup()
            end
            RowCopyButton("$popup_id-$i", name, popup_w)
        end
        ig.EndPopup()
    end

    return result
end

function completion_renderer(item, i, selected)
    clicked = ig.Selectable("##completion", selected)
    ig.SameLine(0, 0)
    ig.Text(item)
    return clicked
end

find_separator(s) = @something(findfirst(':', s), findfirst('.', s), Some(nothing))

# Split a "TOPIC//device" string into (topic, device). Returns ("", s) if no
# topic prefix is present.
function split_topic(s)
    m = match(r"^(\w+)//(.+)$", s)
    isnothing(m) ? ("", s) : (m.captures[1], m.captures[2])
end

# Strip a trailing ":channel" pipeline suffix from a source, leaving the device.
strip_channel(s) = (i = findfirst(':', s); isnothing(i) ? s : s[1:i-1])

# The pipeline channel of a source ("output" for "device:output"), or "" if the
# source names a slow device.
source_channel(s) = (i = findfirst(':', s); isnothing(i) ? "" : s[i+1:end])

# Resolve the (topic, device) schema key for a source string, honouring an
# explicit "topic//" prefix and otherwise looking the device up in the source
# list. Returns nothing if the device isn't known.
function source_key(client, source)
    topic, device = split_topic(strip_channel(source))
    if isempty(topic)
        idx = findfirst(s -> s.name == device, client.source_list)
        isnothing(idx) ? nothing : (client.source_list[idx].topic, device)
    else
        (topic, device)
    end
end

# DeviceProperties for a source, honouring an explicit "topic//" prefix so the
# schema is fetched from the right topic for ambiguous device names.
function source_device_props(client, source)
    key = source_key(client, source)
    if isnothing(key)
        DeviceProperties()
    else
        get_source_properties(client, key[1], key[2])
    end
end

# Whether the device schema for `source` has actually arrived, as opposed to
# still being in flight or never requested. Does not trigger a fetch.
function source_schema_loaded(client, source)
    key = source_key(client, source)
    if isnothing(key)
        false
    else
        haskey(client.source_properties, key) && !haskey(client.device_schema_requests, key)
    end
end

# Seed the editor's working fields from the current source string. `text` is a
# composed "device.prop" / "device:out[path]@proxy" string, or (device_only) a
# "topic//device" device name.
function seed_karabo_editor!(dep_state::KaraboDepTextState, text, device_only)
    if device_only || isempty(text)
        dep_state.source = text
        dep_state.property = ""
        dep_state.proxy = ""
        dep_state.proxy_expanded = false
        return
    end

    dep = try
        karabo_dependency(text)
    catch
        nothing
    end
    dep_state.source = isnothing(dep) ? text : dep.source
    dep_state.property = isnothing(dep) ? "" : @something(dep.property, "")
    dep_state.proxy = (isnothing(dep) || isnothing(dep.proxy)) ? "" : dep.proxy
    dep_state.proxy_expanded = !isempty(dep_state.proxy)
end

# Compose the working fields back into a source string. device_only returns the
# bare source (a device name); otherwise "device.prop" / "device:out[path]@proxy".
function compose_karabo_source(dep_state::KaraboDepTextState)
    if dep_state.device_only
        return dep_state.source
    end
    proxy = isempty(dep_state.proxy) ? nothing : dep_state.proxy
    return karabo_dep_string(nothing, dep_state.source, dep_state.property, proxy)
end

# Run the remap for a composed source `raw`; returns the resolved source, or
# nothing if it parked an async device-property lookup (recorded on dep_state)
# for a later frame to pick up.
function resolve_remap!(dep_state::KaraboDepTextState, client::ClientState, raw)
    new_source, pending = remap_source(client, raw, dep_state.proxy_property)
    if isnothing(pending)
        dep_state.proxy_property[] = nothing
        new_source
    else
        dep_state.pending_remap_id = pending
        dep_state.pending_remap_source = raw
        nothing
    end
end

"""
    KaraboDepText(label, text, dep_state, source_list, client)
        -> (edited::Bool, new_text::String)

Editable widget for a Karabo source. Renders inline as a read-only, elided field
showing the current value; clicking it opens a floating editor window with
separate Source / Property / (optional) Proxy fields, each with its own
autocompletion. The composed value is committed only when the window's OK button
is pressed.

`device_only` restricts the window to just the Source field (for
`Parameter{KaraboDevice}`); `allow_slow=false` hides slow-property completion.
The window itself is drawn once at the top level of the frame by
`draw_karabo_editor`; this function only records the request and picks up the
committed result (running the remap) on the following frame.
"""
function KaraboDepText(label, text, dep_state::KaraboDepTextState,
                       source_list, client::ClientState;
                       device_only::Bool=false, allow_slow::Bool=true,
                       focus::Bool=false)
    # If a previous commit kicked off an async remap, either resolve it now or
    # show a disabled spinner placeholder until the request lands.
    if !isnothing(dep_state.pending_remap_id)
        if !is_pending(client, dep_state.pending_remap_id)
            raw = dep_state.pending_remap_source
            dep_state.pending_remap_id = nothing
            dep_state.pending_remap_source = nothing
            resolved = resolve_remap!(dep_state, client, raw)
            if !isnothing(resolved)
                return true, resolved
            end
        end
        @Disabled true ig.Text(@something(dep_state.pending_remap_source, ""))
        ig.SameLine()
        Spinner("Resolving proxy...")
        return false, text
    end

    # The editor window set `committed` last frame: run the remap and return it.
    if !isnothing(dep_state.committed)
        raw = dep_state.committed
        dep_state.committed = nothing
        if device_only
            return true, raw
        end
        resolved = resolve_remap!(dep_state, client, raw)
        if isnothing(resolved)
            return false, text
        else
            return true, resolved
        end
    end

    # Read-only, clickable display of the current value.
    clicked = KaraboDepDisplay(label, text)
    if clicked || focus
        seed_karabo_editor!(dep_state, text, device_only)
        dep_state.label = label
        dep_state.source_list = source_list
        dep_state.device_only = device_only
        dep_state.allow_slow = allow_slow
        dep_state.trigger = true
        dep_state.frames = 0
        client.karabo_editor = dep_state
    end

    return false, text
end

# A framed, clickable read-only field that looks like a text input and shows the
# current (elided) source with a trailing edit icon. Returns true when clicked.
function KaraboDepDisplay(label, text)
    max_chars = 30
    empty = isempty(text)
    display = empty ? "set source\u2026" : (length(text) > max_chars ? text[1:max_chars] * "\u2026" : text)

    icon = "\uf044"  # pencil
    icon_w = ig.CalcTextSize(icon).x
    padding = ImVec2(4, 2)
    min_width = ig.CalcTextSize("m").x * 13
    text_w = ig.CalcTextSize(display).x
    width = max(text_w + icon_w + 8, min_width) + 2 * padding.x

    ig.InvisibleButton("##karabo-display-$(label)", ImVec2(width, ig.GetFontSize() + 2 * padding.y))
    hovered = ig.IsItemHovered()
    clicked = ig.IsItemClicked()

    draw_list = ig.GetWindowDrawList()
    p_min = ig.GetItemRectMin()
    p_max = ig.GetItemRectMax()
    ig.AddRectFilled(draw_list, p_min, p_max,
                     ig.GetColorU32(hovered ? ig.ImGuiCol_FrameBgHovered : ig.ImGuiCol_FrameBg),
                     unsafe_load(ig.GetStyle().FrameRounding))
    ig.AddRect(draw_list, p_min, p_max, ig.GetColorU32(ig.ImGuiCol_Border),
               unsafe_load(ig.GetStyle().FrameRounding))
    text_col = ig.GetColorU32(empty ? ig.ImGuiCol_TextDisabled : ig.ImGuiCol_Text)
    ig.AddText(draw_list, ImVec2(p_min.x + padding.x, p_min.y + padding.y), text_col, display)
    ig.AddText(draw_list, ImVec2(p_max.x - icon_w - padding.x, p_min.y + padding.y),
               ig.GetColorU32(ig.ImGuiCol_TextDisabled), icon)

    if hovered && length(text) > max_chars
        node_tooltip(text)
    end

    return clicked
end

# Signed distance from a point to a rounded rectangle centered at the origin.
function rounded_box_sdf(px, py, hx, hy, r)
    qx = abs(px) - (hx - r)
    qy = abs(py) - (hy - r)
    return sqrt(max(qx, 0f0)^2 + max(qy, 0f0)^2) + min(max(qx, qy), 0f0) - r
end

# Bake the drop-shadow texture: black with an alpha = erf(distance) falloff around
# a rounded rect (a Gaussian falloff, exact for straight edges), laid out with a
# 2px flat centre so it 9-slices cleanly. The corner cell (blur margin + corner
# radius) maps onto the window corner; the central strip stretches along the edges.
function build_window_shadow()
    sigma = 10f0
    corner = unsafe_load(ig.GetStyle().WindowRounding)
    margin = ceil(Int, 3 * sigma)
    cell = margin + ceil(Int, corner)
    size = 2 * cell + 2                     # +2 → a 2px flat centre strip
    hs = Float32(cell - margin) + 1f0       # rect half-size: corner radius + 1px
    center = (size - 1) / 2f0
    inv = 1f0 / (sqrt(2f0) * sigma)

    pixels = Vector{UInt8}(undef, 4 * size * size)
    idx = 1
    for y in 0:size-1, x in 0:size-1
        d = rounded_box_sdf(x - center, y - center, hs, hs, max(corner, 0f0))
        cov = clamp(0.5f0 - 0.5f0 * base_erf(d * inv), 0f0, 1f0)
        pixels[idx] = 0x00
        pixels[idx+1] = 0x00
        pixels[idx+2] = 0x00
        pixels[idx+3] = round(UInt8, cov * 255)
        idx += 4
    end

    tex_ref = Ref{GLuint}(0)
    glGenTextures(1, tex_ref)
    tex = tex_ref[]
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)

    return WindowShadow(tex, size, cell, Float32(margin), corner)
end

# 9-slice the baked shadow texture around the window rect [win_min, win_max].
function draw_window_shadow(shadow::WindowShadow, win_min::ImVec2, win_max::ImVec2)
    tex = ig.ImTextureRef(ig.ImTextureID(shadow.tex))
    m = shadow.margin
    r = shadow.corner
    u1 = shadow.cell / shadow.size
    u2 = (shadow.size - shadow.cell) / shadow.size
    col = ig.IM_COL32(255, 255, 255, 140)

    # Screen split points: outer shadow bounds, then the window corner radius.
    sx0 = win_min.x - m; sx3 = win_max.x + m
    sy0 = win_min.y - m; sy3 = win_max.y + m
    sx1 = win_min.x + r; sx2 = win_max.x - r
    sy1 = win_min.y + r; sy2 = win_max.y - r

    dl = ig.GetWindowDrawList()
    patch(x0, y0, x1, y1, uu0, vv0, uu1, vv1) =
        ig.AddImage(dl, tex, ImVec2(x0, y0), ImVec2(x1, y1), ImVec2(uu0, vv0), ImVec2(uu1, vv1), col)
    ig.PushClipRectFullScreen(dl)
    patch(sx0, sy0, sx1, sy1, 0, 0, u1, u1)     # top-left
    patch(sx2, sy0, sx3, sy1, u2, 0, 1, u1)     # top-right
    patch(sx0, sy2, sx1, sy3, 0, u2, u1, 1)     # bottom-left
    patch(sx2, sy2, sx3, sy3, u2, u2, 1, 1)     # bottom-right
    patch(sx1, sy0, sx2, sy1, u1, 0, u2, u1)    # top
    patch(sx1, sy2, sx2, sy3, u1, u2, u2, 1)    # bottom
    patch(sx0, sy1, sx1, sy2, 0, u1, u1, u2)    # left
    patch(sx2, sy1, sx3, sy2, u2, u1, 1, u2)    # right
    ig.PopClipRect(dl)
end

# Draw the Karabo source editor window. Called once at the top level of the
# frame; `client.karabo_editor` names the open widget's state (or nothing). The
# window commits into `dep_state.committed` on OK and clears `karabo_editor` when
# it closes (OK, Cancel, Escape, or losing focus).
function draw_karabo_editor(client::ClientState)
    dep_state = client.karabo_editor
    isnothing(dep_state) && return

    title = "Edit Karabo source##karabo-editor-$(dep_state.label)"
    if dep_state.trigger
        viewport = unsafe_load(ig.igGetMainViewport())
        center = ImVec2(viewport.Pos.x + viewport.Size.x / 2, viewport.Pos.y + viewport.Size.y / 2)
        ig.SetNextWindowPos(center, ig.ImGuiCond_Appearing, ImVec2(0.5, 0.5))
        ig.SetNextWindowSize(ImVec2(460, dep_state.device_only ? 150 : 260), ig.ImGuiCond_Appearing)
        ig.SetNextWindowFocus()
        dep_state.trigger = false
    end

    flags = ig.ImGuiWindowFlags_NoCollapse | ig.ImGuiWindowFlags_NoDocking |
            ig.ImGuiWindowFlags_NoSavedSettings

    commit = false
    cancel = false
    if ig.Begin(title, C_NULL, flags)
        dep_state.frames += 1
        dep_state.ac_hovered = false
        dep_state.ac_active = false

        # Soft drop shadow to lift the (non-modal) editor off the app. Drawn on
        # the window's own draw list, which is topmost while the editor is open.
        win_min = ig.GetWindowPos()
        win_size = ig.GetWindowSize()
        win_max = ImVec2(win_min.x + win_size.x, win_min.y + win_size.y)
        draw_window_shadow(state[].window_shadow, win_min, win_max)

        ig.TextDisabled("Source")
        karabo_source_field(dep_state, client)

        if !dep_state.device_only
            ig.TextDisabled("Property")
            karabo_property_field(dep_state, client)

            karabo_proxy_field(dep_state, client)
        end

        ig.Separator()
        preview = compose_karabo_source(dep_state)
        ig.TextWrapped("Preview: " * (isempty(preview) ? "\u2014" : preview))
        ig.Dummy(0, 4)

        if ig.Button("Cancel")
            cancel = true
        end
        ig.SameLine()
        @Disabled !karabo_source_valid(dep_state) begin
            if ig.Button("OK")
                commit = true
            end
        end

        # Enter confirms, like clicking OK. Skipped when a completion popup is
        # open (it captures Enter to pick a row) or the source isn't valid yet.
        if ig.IsKeyPressed(ig.ImGuiKey_Enter) && !dep_state.ac_active && karabo_source_valid(dep_state)
            commit = true
        end

        if ig.IsKeyPressed(ig.ImGuiKey_Escape)
            cancel = true
        end

        # Close when focus leaves the window (and its autocomplete popups),
        # except on the opening frame before focus has settled.
        focused = ig.IsWindowFocused(ig.ImGuiFocusedFlags_RootAndChildWindows)
        lost_focus = dep_state.frames > 1 && !focused && !dep_state.ac_hovered

        if commit
            dep_state.committed = compose_karabo_source(dep_state)
            client.karabo_editor = nothing
        elseif cancel || lost_focus
            client.karabo_editor = nothing
        end
    end
    ig.End()
end

# A Karabo source is valid to commit when it has a source; non-device-only
# sources additionally require a property.
function karabo_source_valid(dep_state::KaraboDepTextState)
    isempty(dep_state.source) && return false
    dep_state.device_only && return true
    return !isempty(dep_state.property)
end

# Source input with the device/channel tree completion. Selecting a device row
# fills a slow source; selecting a nested channel fills a "device:channel" fast
# source.
function karabo_source_field(dep_state::KaraboDepTextState, client::ClientState)
    popup_id = "karabo-src-$(dep_state.label)"
    field_state = get!(ElidedTextState, elided_text_states, ig.GetID("##$(popup_id)-state"))

    ig.SetNextItemWidth(-1)
    _, new_text = SafeInputText("##$(popup_id)"; current_text=dep_state.source, hint="device or device:output")
    dep_state.source = new_text
    if ig.IsItemActive()
        field_state.edit = ElidedEditState_Edit
    end

    result = nothing
    if field_state.edit != ElidedEditState_NoEdit
        result, hovered = draw_source_completions(popup_id, field_state, dep_state.source,
                                                  dep_state.source_list, dep_state.allow_slow, client)
        dep_state.ac_hovered |= hovered
        dep_state.ac_active = true
        if !ig.IsItemActive() && !hovered
            field_state.edit = ElidedEditState_NoEdit
        end
    end
    if !isnothing(result)
        dep_state.source = result
        field_state.edit = ElidedEditState_NoEdit
        # Re-check the property against the newly picked source (deferred until
        # its schema arrives); device_only sources have no property field.
        if !dep_state.device_only
            dep_state.revalidate_property = true
        end
        # Selecting from the popup moved focus away; return it to the editor so
        # the focus-loss close doesn't fire.
        ig.SetWindowFocus()
    end
end

# The source string for a device, topic-qualified when its name is ambiguous.
source_base(dev::SourceInfo) = dev.ambiguous ? "$(dev.topic)//$(dev.name)" : dev.name

# Draw the source completion popup: a fuzzy-matched device list on the left, and
# the selected device's pipeline channels in a column on the right (menu style,
# so the device rows don't move as the selection does). Device schemas are
# fetched lazily (only for the visible rows, and only after the query has been
# stable for 1s) so typing doesn't flood the webproxy. Returns (selected, hovered).
function draw_source_completions(popup_id, field_state::ElidedTextState, text::AbstractString,
                                 source_list::Vector{SourceInfo}, allow_slow::Bool,
                                 client::ClientState)
    topic_fixed, query = split_topic(strip_channel(text))
    sources = isempty(topic_fixed) ? source_list :
              filter(s -> s.topic == topic_fixed, source_list)

    now = ig.GetTime()
    if text != field_state.last_query
        field_state.last_query = text
        field_state.last_change_time = now
        field_state.channel_idx = 0
    end
    debounced = now - field_state.last_change_time >= 1.0

    scored = fuzzy_match(query, sources, s -> s.name)
    isempty(scored) && return nothing, false

    max_rows = 8
    visible = @view scored[1:min(length(scored), max_rows)]

    # Menu-style navigation: Up/Down move within the current column, Right steps
    # into the channel column and Left back out. `channel_idx == 0` means the
    # device row itself is the current row; both indices are clamped below, once
    # the selected device (and so its channel count) is known.
    down = ig.IsKeyPressed(ig.ImGuiKey_DownArrow)
    up = ig.IsKeyPressed(ig.ImGuiKey_UpArrow)
    if field_state.channel_idx == 0
        if down
            field_state.selected_idx += 1
        elseif up
            field_state.selected_idx -= 1
        end
        if ig.IsKeyPressed(ig.ImGuiKey_RightArrow)
            field_state.channel_idx = 1
        end
    else
        if down
            field_state.channel_idx += 1
        elseif up
            field_state.channel_idx = max(field_state.channel_idx - 1, 1)
        end
        if ig.IsKeyPressed(ig.ImGuiKey_LeftArrow)
            field_state.channel_idx = 0
        end
    end
    enter = ig.IsKeyPressed(ig.ImGuiKey_Tab) || ig.IsKeyPressed(ig.ImGuiKey_Enter)

    # The channel column follows the selection with a frame of lag: hovering a row
    # moves the selection only once the device list has been drawn, and the popup
    # is sized here, before that.
    field_state.selected_idx = clamp(field_state.selected_idx, 1, length(visible))
    selected_dev = visible[field_state.selected_idx][2]
    selected_props = get(client.source_properties,
                         (selected_dev.topic, selected_dev.name), nothing)
    channels = isnothing(selected_props) ? String[] : sort!(collect(keys(selected_props.fast)))
    field_state.channel_idx = clamp(field_state.channel_idx, 0, length(channels))

    # Both columns are sized to their longest row, and the channel column is only
    # there when the current device actually has fast sources.
    style = ig.GetStyle()
    window_padding = unsafe_load(style.WindowPadding)
    item_spacing_x = unsafe_load(style.ItemSpacing.x)
    marker_width = ig.CalcTextSize(" > ").x

    # Device row: "name (topic)" with the marker right-aligned after it.
    device_width = marker_width + item_spacing_x + maximum(visible) do scored
        dev = scored[2]
        ig.CalcTextSize(dev.name).x + item_spacing_x + ig.CalcTextSize("($(dev.topic))").x
    end

    max_rows_shown = 12
    rows = clamp(max(length(visible), length(channels) + 1), 1, max_rows_shown)

    # The channel column is a bordered child, so it carries its own padding, plus
    # a scrollbar once its channels don't all fit.
    channel_width = 0f0
    channel_height = 0f0
    if !isempty(channels)
        widest = max(ig.CalcTextSize("fast data").x,
                     maximum(ch -> ig.CalcTextSize(":$(ch)").x, channels))
        scrollbar = length(channels) + 1 > max_rows_shown ? unsafe_load(style.ScrollbarSize) : 0f0
        channel_width = item_spacing_x + widest + 2 * window_padding.x + scrollbar
        channel_height = 2 * window_padding.y
    end

    input_min, input_max = editor_item_rect()
    row_height = ig.GetTextLineHeightWithSpacing()
    popup_width = 2 * window_padding.x + device_width + channel_width
    popup_height = rows * row_height + 2 * window_padding.y + channel_height
    ig.SetNextWindowPos(ImVec2(input_min.x, input_max.y))
    ig.SetNextWindowSize(ImVec2(popup_width, popup_height))

    flags = ig.ImGuiWindowFlags_NoTitleBar | ig.ImGuiWindowFlags_NoResize |
            ig.ImGuiWindowFlags_NoMove | ig.ImGuiWindowFlags_NoFocusOnAppearing |
            ig.ImGuiWindowFlags_NoSavedSettings | ig.ImGuiWindowFlags_Tooltip

    result = nothing
    hovered = false
    if ig.Begin("##autocomplete-$(popup_id)", C_NULL, flags)
        # The editor window is focused, so it sits ahead of this never-focused
        # popup in the hit-test order and would steal hover where they overlap.
        # Bring the popup to the front of that order (without taking keyboard
        # focus) so its rows hover/click normally.
        ig.BringWindowToDisplayFront(ig.GetCurrentWindow())
        hovered = ig.IsWindowHovered(ig.ImGuiHoveredFlags_ChildWindows) ||
                  ig.IsWindowFocused(ig.ImGuiFocusedFlags_RootAndChildWindows)

        # Right-aligned position for the per-row spinner / fast-source marker.
        marker_x = device_width - marker_width

        if ig.BeginChild("##devices-$(popup_id)", ImVec2(device_width, 0))
            for (idx, (_, dev)) in enumerate(visible)
                key = (dev.topic, dev.name)
                in_flight = haskey(client.device_schema_requests, key)
                props = get(client.source_properties, key, nothing)
                # Trigger a lazy schema fetch once the query has settled.
                if debounced && isnothing(props) && !in_flight
                    get_source_properties(client, dev.topic, dev.name)
                end
                has_channels = !isnothing(props) && !isempty(props.fast)

                # A device row is a slow source when those are allowed; otherwise
                # picking it just steps into its channels, which are the only
                # valid choice.
                current = idx == field_state.selected_idx && field_state.channel_idx == 0
                picked = ig.Selectable("$(dev.name)##dev-$idx", current)
                if ig.IsItemHovered()
                    field_state.selected_idx = idx
                    field_state.channel_idx = 0
                end
                if current && enter
                    picked = true
                end
                if picked
                    if allow_slow
                        result = source_base(dev)
                    elseif has_channels
                        field_state.channel_idx = 1
                    end
                end
                ig.SameLine()
                ig.TextDisabled("($(dev.topic))")

                # Right-aligned marker: a spinner while the schema loads, then a
                # '>' for devices that turned out to have fast sources.
                if in_flight
                    ig.SameLine(marker_x)
                    Spinner()
                elseif has_channels
                    ig.SameLine(marker_x)
                    ig.TextDisabled(">")
                end
            end
        end
        ig.EndChild()

        # The channel column of the selected device, drawn only when it has any.
        # A lighter background and a rounded border set it apart from the device
        # list as its own menu.
        if !isempty(channels)
            ig.SameLine()
            ig.PushStyleColor(ig.ImGuiCol_ChildBg, ig.GetColorU32(ig.ImGuiCol_MenuBarBg))
            ig.PushStyleVar(ig.ImGuiStyleVar_ChildRounding, unsafe_load(style.PopupRounding))
            if ig.BeginChild("##channels-$(popup_id)", ImVec2(0, 0), ig.ImGuiChildFlags_Borders)
                ig.TextDisabled("fast data")
                for (ci, ch) in enumerate(channels)
                    current = ci == field_state.channel_idx
                    picked = ig.Selectable(":$(ch)##ch-$ci", current)
                    if current && enter
                        picked = true
                    end
                    if picked
                        result = "$(source_base(selected_dev)):$(ch)"
                    end
                    if ig.IsItemHovered()
                        field_state.channel_idx = ci
                    end
                    # Keep the keyboard selection in view when the channel list
                    # is long enough to scroll.
                    if current && (down || up)
                        ig.SetScrollHereY()
                    end
                end
            end
            ig.EndChild()
            ig.PopStyleVar()
            ig.PopStyleColor()
        end
    end
    ig.End()

    return result, hovered
end

# Property input, completing slow properties (bare device source) or the fast
# data paths of the selected channel (device:channel source).
function karabo_property_field(dep_state::KaraboDepTextState, client::ClientState)
    popup_id = "karabo-prop-$(dep_state.label)"
    field_state = get!(ElidedTextState, elided_text_states, ig.GetID("##$(popup_id)-state"))

    props = source_device_props(client, dep_state.source)
    channel = source_channel(dep_state.source)
    names = if isempty(channel)
        dep_state.allow_slow ? props.slow.names : String[]
    else
        get(props.fast, channel, PropertyList()).names
    end

    # After a new source is picked, drop a property that doesn't belong to it and
    # focus the field so a replacement can be typed. Deferred until the schema
    # arrives so a property that is still valid is kept.
    if dep_state.revalidate_property && source_schema_loaded(client, dep_state.source)
        if !karabo_property_valid(client, dep_state.source, dep_state.property, names)
            dep_state.property = ""
            dep_state.property_focus = true
        end
        dep_state.revalidate_property = false
    end

    if dep_state.property_focus
        ig.SetKeyboardFocusHere()
        dep_state.property_focus = false
    end
    ig.SetNextItemWidth(-1)
    _, new_text = SafeInputText("##$(popup_id)"; current_text=dep_state.property)
    dep_state.property = new_text
    input_min, input_max = editor_item_rect()
    if ig.IsItemActive()
        field_state.edit = ElidedEditState_Edit
    end

    if field_state.edit != ElidedEditState_NoEdit && !isempty(names)
        ac = CompletionResult(names, completion_renderer, new_text,
                              "karabo-prop:$(dep_state.source)")
        result, hovered = draw_autocomplete_popup(popup_id, field_state, ac,
                                                  input_min, input_max)
        dep_state.ac_hovered |= hovered
        dep_state.ac_active = true
        if !ig.IsItemActive() && !hovered
            field_state.edit = ElidedEditState_NoEdit
        end
        if !isnothing(result)
            dep_state.property = result
            field_state.edit = ElidedEditState_NoEdit
            ig.SetWindowFocus()
        end
    end
end

# Optional proxy field, collapsed behind a "+ proxy device" button until used.
# The proxy is itself a device:output source, so it completes against the same
# device/channel tree as the source field (fast sources only, no slow devices).
function karabo_proxy_field(dep_state::KaraboDepTextState, client::ClientState)
    if !dep_state.proxy_expanded && isempty(dep_state.proxy)
        if ig.SmallButton("+ proxy device##proxy-$(dep_state.label)")
            dep_state.proxy_expanded = true
        end
        return
    end

    popup_id = "karabo-proxy-$(dep_state.label)"
    field_state = get!(ElidedTextState, elided_text_states, ig.GetID("##$(popup_id)-state"))

    ig.TextDisabled("Proxy")
    clear_w = ig.GetFrameHeight()
    ig.SetNextItemWidth(-(clear_w + unsafe_load(ig.GetStyle().ItemSpacing.x)))
    _, new_text = SafeInputText("##$(popup_id)"; current_text=dep_state.proxy, hint="device:output")
    dep_state.proxy = new_text
    if ig.IsItemActive()
        field_state.edit = ElidedEditState_Edit
    end

    result = nothing
    if field_state.edit != ElidedEditState_NoEdit
        result, hovered = draw_source_completions(popup_id, field_state, dep_state.proxy,
                                                  dep_state.source_list, false, client)
        dep_state.ac_hovered |= hovered
        dep_state.ac_active = true
        if !ig.IsItemActive() && !hovered
            field_state.edit = ElidedEditState_NoEdit
        end
    end

    ig.SameLine()
    if ig.Button("\uf00d##karabo-proxy-clear-$(dep_state.label)", ImVec2(clear_w, clear_w))
        dep_state.proxy = ""
        dep_state.proxy_expanded = false
        field_state.edit = ElidedEditState_NoEdit
    end

    if !isnothing(result)
        dep_state.proxy = result
        field_state.edit = ElidedEditState_NoEdit
        # Selecting from the popup moved focus away; return it to the editor.
        ig.SetWindowFocus()
    end
end

# Draw a dependency editor widget with a type selector (Karabo/Variable) and
# autocomplete text field. Returns (edited::Bool, new_dep::Dependency) where
# new_dep is the updated dependency if edited.
#
# - `dep`: the current Dependency value
# - `dep_state`: mutable DepTextState tracking the selected type and karabo state
# - `source_list`: Karabo source list for Karabo-mode completions
# - `variable_names`: list of variable names (including subvariable outputs) for Variable-mode completions
# - `device_only`: if true, Karabo mode only edits the device name (no property)
function DepText(label, dep::Dependency, dep_state::DepTextState,
                 source_list, variable_names::Vector{String}, client::ClientState;
                 device_only::Bool=false, variable_name::String="")
    # Type selector. A real combo's dropdown opens mid-node and lands in the wrong
    # place, so draw a button here that records the request and defer the popup to
    # after EndNode under Suspend/Resume (see draw_dag). The popup mutates dep_state
    # and sets wants_focus so the new text field grabs focus next frame.
    kind_label = dep_state.is_karabo ? "Karabo" : "Variable"
    focus = dep_state.wants_focus
    dep_state.wants_focus = false
    if ig.Button("$(kind_label)##dep-kind-$(label)", ImVec2(ig.CalcTextSize("Variable ").x, 0))
        client.dep_kind_popup = ("dep-kind-$(label)", dep_state)
        client.dep_kind_popup_trigger = true
    end

    ig.SameLine()

    if dep_state.is_karabo
        text = dep.kind == DepKind_Karabo ? string(dep) : ""
        focus &= isempty(text)
        edited, new_text = KaraboDepText(label, text, dep_state.karabo_state,
                                         source_list, client;
                                         device_only, focus)
        if edited
            return true, karabo_dependency(new_text)
        end
    else
        text = dep.kind == DepKind_Karabo ? "" : dep.name
        focus &= isempty(text)
        edited, new_text = ElidedText(label, text;
            editable=true, focus,
            completions=input -> begin
                prefix = variable_name * "."
                filtered = filter(v -> v != variable_name && !startswith(v, prefix), variable_names)
                CompletionResult(filtered, completion_renderer, input, "variable")
            end)
        if edited && !isempty(new_text)
            return true, Dependency(new_text)
        end
    end

    return false, dep
end

function variable_name_validator(new_name, current_name)
    client = state[].client

    if isempty(new_name)
        "Name cannot be empty"
    elseif !Meta.isidentifier(new_name)
        "'$(new_name)' is not a valid Julia identifier"
    elseif new_name ∈ client.variable_names && new_name != current_name
        "A variable named '$(new_name)' already exists"
    else
        nothing
    end
end

variable_name_validator(current_name) = Base.Fix2(variable_name_validator, current_name)
