# Apply a source transformation to the loaded context file. `transform` takes
# the current source and returns the modified source (or the same source if
# nothing changed). On change, writes the result to the context file and either
# reloads the context or just updates the in-memory source.
function apply_source_edit(transform, state; reload::Bool=true)
    client = state.client
    source = client.context.source

    if isempty(source)
        @error "No context source available for editing"
        return
    end

    new_source = transform(source)
    if new_source == source
        return
    end

    write_context_file(client, new_source)

    if reload
        load_context(state)
    else
        client.context.source = new_source
    end
end

rename_variable(state, old_name::String, new_name::String) =
    apply_source_edit(s -> edit.replace_variable_name(s, old_name, new_name), state)

function read_context_file(client)
    path = resolve_context_path(client)
    if client.embedded_engine
        read(path, String)
    else
        with_sftp(sftp -> read(path, sftp, String), client)
    end
end

function write_context_file(client, text::String)
    path = resolve_context_path(client)
    if client.embedded_engine
        write(path, text)
    else
        # We write to a temporary file and rename it over the original so that a
        # failed write can't truncate the context file.
        tmp_path = "$(path).$(randstring(8)).tmp"
        with_sftp(client) do sftp
            permissions = stat(path, sftp).permissions & 0o7777
            open(tmp_path, sftp; write=true) do f
                write(f, text)
            end
            chmod(tmp_path, sftp, permissions)
            mv(tmp_path, path, sftp)
        end
    end
end

# Read the context file into the editor. This may go over SFTP so don't block the GUI.
function reload_context_editor(client)
    context_editor = client.context_editor
    context_editor.busy = true
    @guiasync try
        context_editor.pending_text = read_context_file(client)
    finally
        context_editor.busy = false
    end
end

# Replace the editor text, keeping the cursor and scroll position
function set_editor_text(editor::TextEditor, text::String)
    cursor = cte.GetMainCursorPosition(editor)
    first_row = cte.GetFirstVisibleRow(editor)

    # The order matters, SetCursor() scrolls to the cursor unless ScrollToLine() comes after it
    cte.SetText(editor, text)
    cte.SetCursor(editor, cursor)
    cte.ScrollToLine(editor, first_row, cte.Scroll_alignTop)
end

function open_context_editor(client)
    client.context_editor.window_open = true
    reload_context_editor(client)
end

function save_context_editor(client)
    context_editor = client.context_editor
    text = cte.GetText(context_editor.editor)
    context_editor.busy = true
    @guiasync try
        write_context_file(client, text)
    finally
        context_editor.busy = false
    end
end

set_group_param(state, var_name::AbstractString, kwarg_name::AbstractString, new_value::AbstractString; reload::Bool=true) =
    apply_source_edit(s -> edit.replace_constructor_kwarg(s, var_name, kwarg_name, new_value), state;
                      reload)

# Replace a dependency (Karabo or variable) in the source code and reload.
rename_dep(state, variable_name::String, arg_name::String, old_dep::Dependency, new_dep::Dependency) =
    apply_source_edit(s -> edit.replace_dep(s, variable_name, arg_name, new_dep), state)
