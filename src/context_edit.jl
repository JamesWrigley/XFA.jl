using Base.JuliaSyntax: @K_str, parseall, SyntaxNode, children, is_leaf, kind, byte_range


# Resolve a macrocall name child to its @-symbol across JuliaSyntax versions.
# Julia 1.13 turned the K"MacroName" leaf into a K"macro_name" wrapper node
# around an Identifier, so we match on the kind name string rather than a
# K"..." literal (the latter errors at macroexpand time on the wrong version).
function macro_name_symbol(c)
    kc = string(kind(c))
    if kc == "MacroName"        # Julia < 1.13: leaf token, val like Symbol("@Variable")
        return c.val
    elseif kc == "macro_name"   # Julia >= 1.13: wrapper around an Identifier
        cs = children(c)
        return isnothing(cs) || isempty(cs) ? nothing : Symbol("@", cs[1].val)
    else
        return nothing
    end
end


"""
    replace_variable_name(source, old_name, new_name) -> String

Rename a variable in the context source code. Replaces the variable's definition
name and all references to it in other @Variable definitions. A group/input,
whose definition is a `name = Constructor(...)` assignment rather than a
@Variable, is renamed by its assignment name. Returns the modified source, or
the original source unchanged if the variable was not found.
"""
function replace_variable_name(source::String, old_name::String, new_name::String)
    tree = parseall(SyntaxNode, source; ignore_errors=true)

    # Find all identifier leaves with the old name inside @Variable macrocalls
    variable_macros = find_nodes(tree) do node
        kind(node) == K"macrocall" && any(children(node)) do c
            macro_name_symbol(c) == Symbol("@Variable")
        end
    end

    targets = SyntaxNode[]
    for vm in variable_macros
        append!(targets, find_nodes(vm) do node
            is_leaf(node) && kind(node) == K"Identifier" && node.val == Symbol(old_name)
        end)
    end

    # Groups and inputs are defined by assignment, so they have no @Variable to
    # rename; the references above still cover uses inside other variables.
    assign_node = find_assignment_call(tree, old_name)
    if !isnothing(assign_node)
        push!(targets, children(assign_node)[1])
    end

    if isempty(targets)
        @warn "No occurrences of '$(old_name)' found in context file"
        return source
    end

    # Replace in reverse byte order to preserve offsets
    sort!(targets; by=n -> first(byte_range(n)), rev=true)
    for node in targets
        br = byte_range(node)
        source = source[1:first(br)-1] * new_name * source[last(br)+1:end]
    end

    return source
end

# Apply a source transformation to the loaded context file. `transform` takes
# the current source and returns the modified source (or the same source if
# nothing changed). On change, writes the result to the context file and either
# reloads the context or just updates the in-memory source.
function apply_source_edit(state, transform; reload::Bool=true)
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

    if client.embedded_engine
        write(client.context_path, new_source)
    else
        open(client.context_path, client.sftp; write=true) do f
            write(f, new_source)
        end
    end

    if reload
        load_context(state)
    else
        client.context.source = new_source
    end
end

rename_variable(state, old_name::String, new_name::String) =
    apply_source_edit(state, s -> replace_variable_name(s, old_name, new_name))

"""
Find all descendant nodes matching a predicate.
"""
function find_nodes(pred, node::SyntaxNode, results=SyntaxNode[])
    if pred(node)
        push!(results, node)
    end
    if !is_leaf(node)
        for child in children(node)
            find_nodes(pred, child, results)
        end
    end
    return results
end

"""
Find the `@Variable` macrocall node that defines a given variable name.
"""
function find_variable_node(tree::SyntaxNode, var_name::String)
    variable_macros = find_nodes(tree) do node
        kind(node) == K"macrocall" && any(children(node)) do c
            macro_name_symbol(c) == Symbol("@Variable")
        end
    end

    for vm in variable_macros
        vm_children = children(vm)
        isnothing(vm_children) && continue

        for child in vm_children
            # Shorthand form: @Variable name -> karabo"..."
            # Tree: [macrocall @Variable [-> [tuple name] ...]]
            if kind(child) == K"->"
                cs = children(child)
                isnothing(cs) && continue
                tuple_node = cs[1]
                tuple_cs = children(tuple_node)
                if !isnothing(tuple_cs) && !isempty(tuple_cs)
                    first_child = tuple_cs[1]
                    if is_leaf(first_child) && first_child.val == Symbol(var_name)
                        return vm
                    end
                end
            end

            # Call form: @Variable name(arg -> karabo"...") ... end
            # Tree: [macrocall @Variable [call name [-> ...]]]
            if kind(child) == K"call"
                cs = children(child)
                if !isnothing(cs) && !isempty(cs)
                    name_node = cs[1]
                    if is_leaf(name_node) && name_node.val == Symbol(var_name)
                        return vm
                    end
                end
            end

            # Function form: @Variable function name(...) ... end
            # Tree: [macrocall @Variable [function [call name [-> ...]] ...]]
            if kind(child) == K"function"
                cs = children(child)
                if !isnothing(cs) && !isempty(cs)
                    call_node = cs[1]
                    call_cs = children(call_node)
                    if !isnothing(call_cs) && !isempty(call_cs)
                        name_node = call_cs[1]
                        if is_leaf(name_node) && name_node.val == Symbol(var_name)
                            return vm
                        end
                    end
                end
            end
        end
    end

    return nothing
end

# Return the string content inside a Karabo string macro literal, i.e. the
# dependency string without the topic prefix.
karabo_dep_content(dep::Dependency) = karabo_dep_string(nothing, dep.source, dep.property, dep.proxy)

# Convert a Dependency to its source code representation.
function dep_to_source(dep::Dependency)
    if dep.kind == DepKind_Karabo
        content = karabo_dep_content(dep)
        if isnothing(dep.topic)
            "karabo\"$(content)\""
        else
            "karabo\"$(dep.topic)//$(content)\""
        end
    else
        dep.name
    end
end

# Convert a Dependency to source code for use as a group constructor kwarg value.
# Variable deps need to be wrapped in Dependency("...") since they can't
# appear as bare identifiers inside a constructor kwarg.
function parameter_dep_to_source(dep::Dependency)
    if dep.kind == DepKind_Karabo
        dep_to_source(dep)
    else
        "Dependency(\"$(dep.name)\")"
    end
end

# Find the arrow node for a specific argument in a @Variable definition.
function find_arg_arrow(var_node::SyntaxNode, variable_name::String, arg_name::String)
    # LHS leaf symbol of an arrow node, or nothing if not a simple arrow.
    arrow_lhs_sym(node) = begin
        kind(node) == K"->" || return nothing
        cs = children(node)
        isnothing(cs) && return nothing
        lhs = cs[1]
        if kind(lhs) == K"tuple"
            tuple_cs = children(lhs)
            (isnothing(tuple_cs) || isempty(tuple_cs) || !is_leaf(tuple_cs[1])) && return nothing
            tuple_cs[1].val
        elseif is_leaf(lhs)
            lhs.val
        else
            nothing
        end
    end

    # Prefer an arrow whose LHS is the requested arg_name. Fall back to a
    # variable_name match only for the shorthand `@Variable name -> ...` form
    # (where the caller may pass a placeholder arg_name like "data"), and only
    # when no arg_name match exists — otherwise a nested `arg -> karabo"..."`
    # inside the body would lose to the outer arrow.
    by_arg = find_nodes(n -> arrow_lhs_sym(n) == Symbol(arg_name), var_node)
    !isempty(by_arg) && return by_arg[1]

    by_var = find_nodes(n -> arrow_lhs_sym(n) == Symbol(variable_name), var_node)
    return isempty(by_var) ? nothing : by_var[1]
end

"""
    replace_dep(source, variable_name, arg_name, new_dep) -> String

Replace the dependency for a specific argument within a `@Variable` definition
or a group constructor call in the source code. Works for both Karabo and
variable dependencies by replacing the entire RHS of the arrow expression (for
variables) or the kwarg value (for group kwargs). Returns the modified source,
or the original source unchanged if the variable or argument was not found.
"""
function replace_dep(source::String, variable_name::String, arg_name::String, new_dep::Dependency)
    tree = parseall(SyntaxNode, source; ignore_errors=true)

    # Try @Variable definition first
    var_node = find_variable_node(tree, variable_name)
    if !isnothing(var_node)
        arrow = find_arg_arrow(var_node, variable_name, arg_name)
        if isnothing(arrow)
            @warn "No argument '$(arg_name)' found in @Variable definition for '$(variable_name)'"
            return source
        end

        rhs = children(arrow)[2]
        br = byte_range(rhs)
        return source[1:first(br)-1] * dep_to_source(new_dep) * source[last(br)+1:end]
    end

    # Fall back to group constructor kwarg
    new_source = replace_constructor_kwarg(source, variable_name, arg_name,
                                           parameter_dep_to_source(new_dep);
                                           warn=false)
    if new_source == source
        @warn "Could not find @Variable definition or group assignment for '$(variable_name)'"
    end
    return new_source
end

# Find an assignment node `name = SomeConstructor(...)` in the AST.
function find_assignment_call(tree::SyntaxNode, name::String)
    assignments = find_nodes(tree) do node
        kind(node) == K"=" || return false
        cs = children(node)
        (isnothing(cs) || length(cs) < 2) && return false

        lhs = cs[1]
        is_leaf(lhs) && lhs.val == Symbol(name) && kind(cs[2]) == K"call"
    end

    return isempty(assignments) ? nothing : assignments[1]
end

# Replace a keyword argument value in a constructor call assigned to `var_name`.
# Handles patterns like: `my_group = Foo(; x=old_value)`
# If the kwarg doesn't exist, it is appended. If there are no kwargs at all,
# a new parameter section is inserted.
function replace_constructor_kwarg(source::String, var_name::String,
                                   kwarg_name::String, new_value::String;
                                   warn::Bool=true)
    tree = parseall(SyntaxNode, source; ignore_errors=true)
    assign_node = find_assignment_call(tree, var_name)
    if isnothing(assign_node)
        if warn
            @warn "Could not find constructor assignment for '$(var_name)'"
        end
        return source
    end

    call_node = children(assign_node)[2]

    # Find the parameters node (kwargs after ;)
    params_node = nothing
    for c in children(call_node)
        if kind(c) == K"parameters"
            params_node = c
            break
        end
    end

    new_kwarg = "$(kwarg_name)=$(new_value)"

    if isnothing(params_node)
        # No kwargs at all — insert before the closing paren
        call_end = last(byte_range(call_node))
        return source[1:call_end-1] * "; $(new_kwarg)" * source[call_end:end]
    end

    # Find the kwarg matching kwarg_name
    kwarg_node = nothing
    for c in children(params_node)
        if kind(c) == K"=" && !isempty(children(c)) &&
           is_leaf(children(c)[1]) && children(c)[1].val == Symbol(kwarg_name)
            kwarg_node = c
            break
        end
    end

    if isnothing(kwarg_node)
        # Kwarg not present — append it after the existing kwargs
        br = byte_range(params_node)
        return source[1:last(br)] * ", $(new_kwarg)" * source[last(br)+1:end]
    end

    # Replace the entire RHS of the kwarg with the new value
    rhs = children(kwarg_node)[2]
    br = byte_range(rhs)
    return source[1:first(br)-1] * new_value * source[last(br)+1:end]
end

# Format a parameter value as its Julia source representation for embedding as
# a group constructor kwarg value or a Parameter positional argument. Returns
# nothing for value types that aren't persistable to source.
format_param_value(s::String) = "\"$(escape_string(s))\""
format_param_value(roi::RectROI) =
    "RectROI($(roi.corner_x), $(roi.corner_y), $(roi.width), $(roi.height))"
format_param_value(roi::LinearROI) = "LinearROI($(roi.start), $(roi.length); axis=:$(roi.axis))"
format_param_value(x::Union{Integer, AbstractFloat, Bool}) = repr(x)
format_param_value(d::KaraboDevice) = "KaraboDevice(\"$(d.topic)\", \"$(d.name)\")"
format_param_value(v::Vector{<:Union{Integer, AbstractFloat}}) =
    "[" * join((repr(x) for x in v), ", ") * "]"
format_param_value(_) = nothing

# Replace the first positional argument of a `name = Parameter(...)` assignment.
# Used for top-level parameters declared like `roi = Parameter(RectROI())`.
function replace_parameter_value(source::String, var_name::String, new_value::String)
    tree = parseall(SyntaxNode, source; ignore_errors=true)
    assign_node = find_assignment_call(tree, var_name)
    if isnothing(assign_node)
        @warn "Could not find Parameter assignment for '$(var_name)'"
        return source
    end

    call_node = children(assign_node)[2]
    cs = children(call_node)
    # cs[1] is the callee; the first non-parameters child after it is the
    # positional value we want to replace.
    arg_node = nothing
    for c in cs[2:end]
        if kind(c) != K"parameters"
            arg_node = c
            break
        end
    end
    if isnothing(arg_node)
        @warn "Parameter assignment for '$(var_name)' has no positional argument"
        return source
    end

    br = byte_range(arg_node)
    return source[1:first(br)-1] * new_value * source[last(br)+1:end]
end

# Replace a dependency inside a group constructor's keyword argument.
# Handles patterns like: `my_group = Foo(; x=karabo"A/B.prop")`
function replace_group_dep(source::String, group_name::String,
                           kwarg_name::String, new_dep::Dependency)
    replace_constructor_kwarg(source, group_name, kwarg_name,
                              parameter_dep_to_source(new_dep))
end

# Whether a `using`/`import` statement brings the name `mod` itself into scope.
# `using Mod`, `import Mod` and `using X: Mod` do, `using Mod: name` doesn't.
function binds_module(node::SyntaxNode, mod::String)
    sym = Symbol(mod)

    # The name bound by an importpath is its last component (`using A.B` binds B).
    binds(n) = if kind(n) == K"importpath"
        cs = children(n)
        !isnothing(cs) && !isempty(cs) && last(cs).val == sym
    elseif kind(n) == K"as"
        cs = children(n)
        is_leaf(cs[2]) && cs[2].val == sym
    else
        false
    end

    for c in children(node)
        if kind(c) == K":"
            # `using X: a, b` — only the imported names are bound, not X itself
            if any(binds, children(c)[2:end])
                return true
            end
        elseif binds(c)
            return true
        end
    end

    return false
end

# Inject a `using <Module>: <Module>` for the top-level module of a
# module-qualified `origin` (e.g. "XfaEngine.KaraboInput") unless the source
# already brings that module into scope. Bare origins need no import, and every
# context module already imports XfaContext (see load_from_string).
function ensure_import(source::String, origin::String)
    parts = split(origin, ".")
    if length(parts) < 2 || parts[1] == "XfaContext"
        return source
    end
    mod = String(parts[1])

    tree = parseall(SyntaxNode, source; ignore_errors=true)
    imports = find_nodes(n -> kind(n) in (K"using", K"import"), tree)
    if any(n -> binds_module(n, mod), imports)
        return source
    end

    stmt = "using $(mod): $(mod)"
    if isempty(imports)
        return stmt * "\n\n" * source
    else
        # find_nodes walks in document order, so the last match is the last import.
        pos = last(byte_range(last(imports)))
        return source[1:pos] * "\n" * stmt * source[pos+1:end]
    end
end

# Source for a new variable: a @Variable reference to the origin function, with
# each wired dependency passed as an override. Group deps are skipped since the
# origin's own definition supplies them.
function variable_source(spec::VariableSpec, name::String, dep_values)
    overrides = ["$(arg) -> $(dep_to_source(dep))" for (arg, dep) in dep_values
                 if dep.kind != DepKind_Group && !isempty(dep.name)]

    if isempty(overrides)
        "@Variable $(name) -> $(spec.origin)"
    else
        "@Variable $(name) -> $(spec.origin)($(join(overrides, ", ")))"
    end
end

# Source for a new group/input: a constructor call assigned to `name`. Only
# wired dependencies and parameters edited away from the spec's defaults are
# passed as kwargs, the rest are left to the constructor's own defaults.
function group_source(spec::VariableSpec, name::String, dep_values, param_values)
    kwargs = String[]

    for (field, dep) in dep_values
        if !isempty(dep.name)
            push!(kwargs, "$(field)=$(parameter_dep_to_source(dep))")
        end
    end

    for (field, param) in param_values
        if isequal(param.value, spec.group_parameters[field].value)
            continue
        end

        value = format_param_value(param.value)
        if isnothing(value)
            throw(ArgumentError("Cannot represent parameter '$(field)' of '$(name)' " *
                                "(a $(typeof(param.value))) in the context file"))
        end
        push!(kwargs, "$(field)=$(value)")
    end

    if isempty(kwargs)
        "$(name) = $(spec.origin)()"
    else
        "$(name) = $(spec.origin)(; $(join(kwargs, ", ")))"
    end
end

# Append a definition for a new variable/group/input built from `spec` to the
# context source, injecting an import for its origin module if needed.
function add_variable_source(source::String, spec::VariableSpec, name::String,
                             dep_values, param_values)
    decl = if spec.kind == VariableKind_Variable
        variable_source(spec, name, dep_values)
    else
        group_source(spec, name, dep_values, param_values)
    end

    return rstrip(ensure_import(source, spec.origin)) * "\n\n" * decl * "\n"
end

# Delete a variable's definition from the context source: the @Variable
# macrocall, or the `name = Constructor(...)` assignment of a group/input. The
# whole line(s) go, leaving the neighbouring declarations separated by a blank
# line. References to the variable elsewhere are left alone.
function remove_variable_source(source::String, name::String)
    tree = parseall(SyntaxNode, source; ignore_errors=true)

    node = find_variable_node(tree, name)
    if isnothing(node)
        node = find_assignment_call(tree, name)
    end
    if isnothing(node)
        throw(ArgumentError("Could not find a definition for '$(name)' in the context file"))
    end

    br = byte_range(node)
    line_start = something(findprev('\n', source, first(br)), 0) + 1
    line_end = something(findnext('\n', source, last(br)), lastindex(source))

    before = rstrip(source[1:line_start-1], '\n')
    after = lstrip(source[line_end+1:end], '\n')
    if isempty(before)
        return after
    elseif isempty(after)
        return before * "\n"
    else
        return before * "\n\n" * after
    end
end

set_group_param(state, var_name::String, kwarg_name::String, new_value::String; reload::Bool=true) =
    apply_source_edit(state, s -> replace_constructor_kwarg(s, var_name, kwarg_name, new_value);
                      reload)

set_parameter_value(state, var_name::String, new_value::String; reload::Bool=true) =
    apply_source_edit(state, s -> replace_parameter_value(s, var_name, new_value); reload)

# Replace a dependency (Karabo or variable) in the source code and reload.
rename_dep(state, variable_name::String, arg_name::String, old_dep::Dependency, new_dep::Dependency) =
    apply_source_edit(state, s -> replace_dep(s, variable_name, arg_name, new_dep))
