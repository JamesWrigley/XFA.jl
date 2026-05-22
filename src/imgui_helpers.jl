mutable struct SafeInputTextState
    buffer::Vector{UInt8}
    reference_text::String
end

function SafeInputTextState(max_len::Int, reference_text::String)
    buffer = zeros(UInt8, max_len + 1) # Add 1 for the null pointer
    SafeInputTextState(buffer, reference_text)
end

const safe_input_text_cache = Dict{UInt32, SafeInputTextState}()


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

    ig.SetNextWindowPos(ImVec2(ig.GetItemRectMin().x, ig.GetItemRectMax().y))
    popup_flags = ig.ImGuiWindowFlags_NoTitleBar
    popup_flags |= ig.ImGuiWindowFlags_NoMove
    popup_flags |= ig.ImGuiWindowFlags_NoResize
    popup_flags |= ig.ImGuiWindowFlags_ChildWindow

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
    ig.AddRect(draw_list, p_min, p_max, color, 0.0, 0, thickness)
    ig.Text(text)
end

function BoxedText(label, text)
    if ig.BeginChild("##webproxy_error_child", ImVec2(0, 0), true,
                     ig.ImGuiWindowFlags_HorizontalScrollbar)
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

@enum ElidedEditState begin
    ElidedEditState_NoEdit
    ElidedEditState_WantEdit
    ElidedEditState_Edit
end

struct CompletionResult
    items::Any
    formatter::Function
    renderer::Function
    query::String
    source::String
end

mutable struct ElidedTextState
    edit::ElidedEditState
    selected_idx::Int
    cached_query::String
    cached_source::String
    cached_scored::Vector{Tuple{Int, Any}}
end

ElidedTextState() = ElidedTextState(ElidedEditState_NoEdit, 1, "", "", Tuple{Int, Any}[])

const elided_text_states = Dict{UInt32, ElidedTextState}()

"""
    fuzzy_match(query, text) -> (matches::Bool, score::Int)

Simple fuzzy match: checks if all characters in `query` appear in `text` in
order (case-insensitive). Score is based on consecutive matches and early
positions.
"""
function fuzzy_match(query::AbstractString, text::AbstractString)
    q = lowercase(query)
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
Default completion renderer: just renders the text of the completion as a
Selectable.
"""
function default_completion_renderer(completion, idx, is_selected)
    ig.Selectable(completion, is_selected)
end


"""
    draw_autocomplete_popup(label, query, completions, completion_text,
                            completion_renderer) -> Union{Nothing, String}

Draw the autocomplete popup below the current item. Returns the selected
completion text if one was chosen, `nothing` otherwise.

- `completions`: iterable of completion items (any type)
- `completion_text(item) -> String`: extracts the text to match against and to
  return on selection
- `completion_renderer(item, index, is_selected)`: draws a single completion row
"""
function draw_autocomplete_popup(label, state::ElidedTextState, ac::CompletionResult)
    popup_label = "##autocomplete-$(label)"

    scored = if ac.query == state.cached_query && ac.source == state.cached_source && !isempty(state.cached_scored)
        state.cached_scored
    else
        state.cached_query = ac.query
        state.cached_source = ac.source
        state.cached_scored = fuzzy_match(ac.query, ac.items, ac.formatter)
    end

    result = nothing
    popup_hovered = false

    if !isempty(scored)
        # Position popup below the input
        input_min = ig.GetItemRectMin()
        input_max = ig.GetItemRectMax()
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
            popup_hovered = ig.IsWindowHovered() || ig.IsWindowFocused()

            # Keyboard navigation
            if ig.IsKeyPressed(ig.ImGuiKey_DownArrow)
                state.selected_idx = min(state.selected_idx + 1, length(scored))
            elseif ig.IsKeyPressed(ig.ImGuiKey_UpArrow)
                state.selected_idx = max(state.selected_idx - 1, 1)
            end

            for (i, (_, item)) in enumerate(scored)
                is_selected = (i == state.selected_idx)
                formatted = ac.formatter(item)
                ig.PushID(i)
                ig.SetNextItemAllowOverlap()
                if ac.renderer(item, i, is_selected)
                    result = formatted
                end
                RowCopyButton("ac-$i", formatted, popup_width)
                if is_selected && (ig.IsKeyPressed(ig.ImGuiKey_Tab) || ig.IsKeyPressed(ig.ImGuiKey_Enter))
                    result = formatted
                end
                ig.PopID()
            end

            ig.End()
        end
    end

    state.selected_idx = clamp(state.selected_idx, 1, max(length(scored), 1))

    return result, popup_hovered
end

function ElidedText(label::AbstractString, text::AbstractString;
                    max_chars::Int=30, editable::Bool=false,
                    focus::Bool=false,
                    completions=nothing,
                    completion_text::Function=string,
                    completion_renderer::Function=default_completion_renderer,
                    callback=C_NULL, user_data=C_NULL,
                    validator=nothing)
    id = ig.GetID(label)
    state = get!(ElidedTextState, elided_text_states, id)
    if focus && state.edit == ElidedEditState_NoEdit
        state.edit = ElidedEditState_WantEdit
    end

    min_width = ig.CalcTextSize("m").x * 13  # minimum clickable width

    if editable && state.edit != ElidedEditState_NoEdit
        just_started = state.edit == ElidedEditState_WantEdit
        if just_started
            ig.SetKeyboardFocusHere()
            state.edit = ElidedEditState_Edit
            state.selected_idx = 1
        end
        ig.SetNextItemWidth(max(min_width, ig.CalcTextSize(text).x + 40))
        edited, new_text = SafeInputText("##elided-$(label)"; current_text=text, reset=just_started,
                                         callback, user_data, validator)
        lost_focus = !just_started && ig.IsItemDeactivated() && !ig.IsItemActive()

        # Draw autocomplete popup if completions are provided
        ac_result = nothing
        ac_hovered = false
        if !isnothing(completions)
            ac = if completions isa Base.Callable
                completions(new_text)
            else
                CompletionResult(completions, completion_text, completion_renderer, new_text, "")
            end
            ac_result, ac_hovered = draw_autocomplete_popup(label, state, ac)
        end

        if !isnothing(ac_result)
            state.edit = ElidedEditState_NoEdit
            return true, ac_result
        elseif edited
            state.edit = ElidedEditState_NoEdit
            if new_text != text && !isempty(new_text)
                return true, new_text
            end
        elseif ig.IsKeyPressed(ig.ImGuiKey_Escape)
            state.edit = ElidedEditState_NoEdit
        elseif lost_focus && !ac_hovered
            state.edit = ElidedEditState_NoEdit
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
                ig.SetTooltip(text)
            end
            if clicked
                state.edit = ElidedEditState_WantEdit
            end
        else
            ig.Text(display_text)
            if ig.IsItemHovered() && elide
                ig.SetTooltip(text)
            end
        end
    end

    return false, text
end

ElidedText(text::AbstractString; max_chars::Int=30) = ElidedText("", text; max_chars)

function InfoMarker(message::AbstractString, marker::AbstractString="?")
    ig.TextDisabled("[$(marker)]")
    if ig.IsItemHovered() && ig.BeginTooltip()
        ig.Text(message)
        ig.EndTooltip()
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
    ig.SameLine(popup_width - width - unsafe_load(ig.GetStyle().WindowPadding.x) * 2)
    if ig.IsMouseHoveringRect(row_min, ImVec2(row_min.x + popup_width, row_max.y))
        CopyButton(label, copy_text)
    else
        ig.Dummy(ImVec2(width, height))
    end
end

# A combo box where each item has a copy-to-clipboard button. Also shows a copy
# button on the combo preview when hovered. Returns true if the selection
# changed.
function CopyableCombo(label, items, selected_idx::Ref{Cint})
    changed = false
    sel = selected_idx[] + 1
    preview = 1 <= sel <= length(items) ? items[sel] : ""
    button_width, button_height = RowCopyButton_size()

    # AllowOverlap so the copy button overlaid on the preview can receive clicks
    ig.SetNextItemAllowOverlap()
    combo_w = ig.CalcItemWidth() + ig.GetFrameHeight()
    ig.SetNextWindowSizeConstraints(ImVec2(0, 0), ImVec2(combo_w, Cfloat(typemax(Int32))))
    if ig.BeginCombo("##$label", preview)
        popup_w = combo_w
        for (i, name) in enumerate(items)
            ig.SetNextItemAllowOverlap()
            if ig.Selectable("##$label-$i")
                new_idx = i - 1
                if new_idx != selected_idx[]
                    selected_idx[] = new_idx
                    changed = true
                end
            end
            ig.SameLine(0, 0)
            ig.Text(name)
            RowCopyButton("$label-$i", name, popup_w)
        end
        ig.EndCombo()
    end

    # Overlay a copy button inside the combo preview, just left of the dropdown arrow.
    # We move the cursor back into the combo's rect to position the button.
    combo_rect_min = ig.GetItemRectMin()
    combo_rect_max = ig.GetItemRectMax()
    combo_w = combo_rect_max.x - combo_rect_min.x
    combo_h = combo_rect_max.y - combo_rect_min.y
    arrow_w = ig.GetFrameHeight()
    save_cursor = ig.GetCursorPos()
    ig.SameLine(0, 0)
    ig.SetCursorPosX(ig.GetCursorPosX() - arrow_w - button_width - 4)
    ig.SetCursorPosY(ig.GetCursorPosY() + (combo_h - button_height) / 2)
    if ig.IsMouseHoveringRect(combo_rect_min, combo_rect_max)
        CopyButton("$label-preview", preview)
    end

    # Restore the cursor and register the combo's right edge as the line position,
    # so the caller's SameLine() places content after the combo, not the overlay button.
    ig.SetCursorPos(save_cursor)
    ig.SameLine(0, 0)
    ig.SetCursorPosX(combo_rect_min.x - ig.GetWindowPos().x + combo_w)
    ig.NewLine()

    return changed
end

function dep_text_callback(callback_data::Ptr{ig.ImGuiInputTextCallbackData})::Cint
    user_data_ptr = unsafe_load(callback_data.UserData)
    state::KaraboDepTextState = unsafe_pointer_to_objref(user_data_ptr)

    state.cursor_pos = unsafe_load(callback_data.CursorPos)

    if !isnothing(state.wanted_text)
        text = state.wanted_text
        state.wanted_text = nothing
        buf_len = unsafe_load(callback_data.BufTextLen)
        ig.DeleteChars(callback_data, 0, buf_len)
        ig.InsertChars(callback_data, 0, text)
    end

    return 0
end

function source_completion_renderer(item::SourceInfo, i, selected)
    clicked = ig.Selectable("##source-$i", selected)
    ig.SameLine(0, 0)
    ig.Text(item.name)
    ig.SameLine()
    ig.TextDisabled("($(item.topic))")
    return clicked
end

function property_completion_renderer(item, i, selected)
    clicked = ig.Selectable("##prop-$i", selected)
    ig.SameLine(0, 0)
    ig.Text(item)
    return clicked
end

find_separator(s) = @something(findfirst(':', s), findfirst('.', s), Some(nothing))

# Strip a "TOPIC//" prefix from a device name, if present.
strip_topic(s) = (m = match(r"^\w+//(.+)$", s); isnothing(m) ? s : m.captures[1])

# Split a "TOPIC//device" string into (topic, device). Returns ("", s) if no
# topic prefix is present.
function split_topic(s)
    m = match(r"^(\w+)//(.+)$", s)
    isnothing(m) ? ("", s) : (m.captures[1], m.captures[2])
end

# Compute completions for a KaraboDependency text input. Returns
# (items, formatter, query) where items is the list to complete from, formatter
# maps an item to the string to insert, and query is the fuzzy match input.
function dep_completions(input, cursor, source_list, device_props::DeviceProperties;
                         allow_slow::Bool=true)
    sep = find_separator(input)
    cursor_after_sep = !isnothing(sep) && cursor >= 0 && cursor > sep - 1

    if isnothing(sep) || !cursor_after_sep
        raw_query = isnothing(sep) ? input : input[1:sep-1]
        suffix = isnothing(sep) ? "" : input[sep:end]

        # Check for a TOPIC// prefix
        topic_match = match(r"^(\w+)//(.*)$", raw_query)
        if !isnothing(topic_match)
            fixed_topic = topic_match.captures[1]
            query = topic_match.captures[2]
            sources = filter(s -> s.topic == fixed_topic, source_list)
            formatter = item -> "$(item.topic)//$(item.name)$(suffix)"
        else
            query = raw_query
            sources = source_list
            formatter = item -> (item.ambiguous ? "$(item.topic)//$(item.name)" : item.name) * suffix
        end
        return (sources, formatter, query)
    elseif input[sep] == '.'
        if !allow_slow
            return (String[], identity, "")
        end
        dev = @view input[1:sep-1]
        query = @view input[sep+1:end]
        return (device_props.slow.names, prop -> "$(dev).$(prop)", query)
    else
        # After ':', check for bracket to distinguish pipeline vs fast property
        after_colon = @view input[sep+1:end]
        bi = findfirst('[', after_colon)
        if isnothing(bi) || !(cursor >= 0 && cursor > sep + bi - 1)
            # After ':' but before '[': complete pipeline output names
            dev = @view input[1:sep-1]
            query = isnothing(bi) ? after_colon : @view after_colon[1:bi-1]
            pipelines = collect(keys(device_props.fast))
            return (pipelines, prop -> "$(dev):$(prop)", query)
        else
            # After '[': complete fast properties for this pipeline
            pipeline = String(@view after_colon[1:bi-1])
            prefix = @view input[1:sep+bi]
            query = @view after_colon[bi+1:end]
            fast_names = get(device_props.fast, pipeline, PropertyList()).names
            return (fast_names, prop -> "$(prefix)$(prop)]", query)
        end
    end
end

"""
    KaraboDepText(label, text, dep_state, source_list, device_props)
        -> (edited::Bool, new_text::String)

Editable text for KaraboDependency fields with autocompletion. Completions
switch automatically based on cursor position:
- Before any separator: source name completions
- After `.`: slow property completions
- After `:`: pipeline output name completions
- After `:pipeline[`: fast property completions for that pipeline

When a source is selected from completions (no separator in the result), a dot
is appended and the widget stays in edit mode for property entry.

The caller manages the `KaraboDepTextState` and should check `dep_state.device`
after each call to determine which source's `DeviceProperties` to pass on the
next frame.
"""
function KaraboDepText(label, text, dep_state::KaraboDepTextState,
                       source_list, device_props::DeviceProperties,
                       client::ClientState;
                       device_only::Bool=false, allow_slow::Bool=true,
                       focus::Bool=false)
    # If a previous edit kicked off an async remap, either resolve it now or
    # show a disabled spinner placeholder until the request lands.
    if !isnothing(dep_state.pending_remap_id)
        if !is_pending(client, dep_state.pending_remap_id)
            pending_source = dep_state.pending_remap_source
            dep_state.pending_remap_id = nothing
            dep_state.pending_remap_source = nothing
            new_source, pending = remap_source(client, pending_source, dep_state.proxy_property)

            if isnothing(pending)
                dep_state.proxy_property[] = nothing
                return true, new_source
            end

            dep_state.pending_remap_id = pending
            dep_state.pending_remap_source = pending_source
        end
        @Disabled true ig.Text(@something(dep_state.pending_remap_source, ""))
        ig.SameLine()
        Spinner("Resolving proxy...")
        return false, text
    end

    id = ig.GetID(label)

    cb = @cfunction(dep_text_callback, Cint, (Ptr{ig.ImGuiInputTextCallbackData},))

    live_text = Ref(text)
    edited, new_text = ElidedText(label, text;
        editable=true, focus,
        callback=cb,
        user_data=pointer_from_objref(dep_state),
        completions=input -> begin
            live_text[] = input
            cursor = device_only ? -1 : dep_state.cursor_pos
            items, formatter, query = dep_completions(input, cursor,
                                                      source_list, device_props;
                                                      allow_slow)
            is_source_list = items isa Vector{SourceInfo}
            renderer = is_source_list ? source_completion_renderer : property_completion_renderer
            mode = is_source_list ? "devices" : "properties"
            source = "karabo:$(mode):" * @something(dep_state.device, "")
            CompletionResult(items, formatter, renderer, query, source)
        end)

    # Update the device field based on current text and cursor position
    sep_idx = find_separator(live_text[])
    cur = dep_state.cursor_pos
    if !isnothing(sep_idx) && cur >= 0 && cur > sep_idx - 1
        dep_state.device = strip_topic(live_text[][1:sep_idx-1])
    else
        dep_state.device = nothing
    end

    if edited
        sep = find_separator(new_text)
        has_colon = !isnothing(sep) && new_text[sep] == ':'
        # A complete expression is either "device.property" (dot separator) or
        # "device:pipeline[property]" (colon separator with closing bracket).
        is_complete = device_only || !isnothing(sep) && (!has_colon || endswith(new_text, ']'))
        if is_complete
            dep_state.cursor_pos = -1
            dep_state.device = nothing
            new_source, pending = remap_source(client, new_text, dep_state.proxy_property)
            if !isnothing(pending)
                dep_state.pending_remap_id = pending
                dep_state.pending_remap_source = new_text
                return false, text
            end
            dep_state.proxy_property[] = nothing
            return true, new_source
        else
            # Incomplete — stay in edit mode (e.g. source or pipeline selected)
            if has_colon && !occursin('[', new_text)
                new_text = new_text * "["
            end
            dep_state.wanted_text = new_text
            dep_state.device = strip_topic(isnothing(sep) ? new_text : new_text[1:sep-1])
            get!(ElidedTextState, elided_text_states, id).edit = ElidedEditState_WantEdit
            return false, text
        end
    end

    # Clean up state when editing is dismissed
    if get!(ElidedTextState, elided_text_states, id).edit == ElidedEditState_NoEdit
        dep_state.cursor_pos = -1
        dep_state.device = nothing
    end

    return false, text
end

function variable_completion_renderer(item, i, selected)
    clicked = ig.Selectable("##var-$i", selected)
    ig.SameLine(0, 0)
    ig.Text(item)
    return clicked
end

# Draw a dependency editor widget with a type selector (Karabo/Variable) and
# autocomplete text field. Returns (edited::Bool, new_dep::Dependency) where
# new_dep is the updated dependency if edited.
#
# - `dep`: the current Dependency value
# - `dep_state`: mutable DepTextState tracking the selected type and karabo state
# - `source_list`: Karabo source list for Karabo-mode completions
# - `device_props`: DeviceProperties for the currently-entered Karabo device
# - `variable_names`: list of variable names (including subvariable outputs) for Variable-mode completions
# - `device_only`: if true, Karabo mode only completes device names (no property)
function DepText(label, dep::Dependency, dep_state::DepTextState,
                 source_list, device_props::DeviceProperties,
                 variable_names::Vector{String}, client::ClientState;
                 device_only::Bool=false, variable_name::String="")
    # Type selector combo
    dep_kinds = ["Karabo", "Variable"]
    current_idx = dep_state.is_karabo ? 0 : 1
    focus = false
    frame_padding = unsafe_load(ig.GetStyle().FramePadding.x)
    ig.SetNextItemWidth(ig.CalcTextSize("Variable ").x)
    if ig.BeginCombo("##dep-kind-$(label)", dep_kinds[current_idx + 1], ig.ImGuiComboFlags_NoArrowButton)
        for (i, kind_label) in enumerate(dep_kinds)
            selected = (i - 1) == current_idx
            if ig.Selectable(kind_label, selected)
                focus = dep_state.is_karabo != (i == 1)
                dep_state.is_karabo = (i == 1)
                # Reset karabo state when switching
                dep_state.karabo_state.cursor_pos = -1
                dep_state.karabo_state.device = nothing
                dep_state.karabo_state.wanted_text = nothing
            end
        end
        ig.EndCombo()
    end

    ig.SameLine()

    if dep_state.is_karabo
        text = dep.kind == DepKind_Karabo ? string(dep) : ""
        focus &= isempty(text)
        edited, new_text = KaraboDepText(label, text, dep_state.karabo_state,
                                         source_list, device_props, client;
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
                CompletionResult(filtered, identity, variable_completion_renderer, input, "variable")
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
