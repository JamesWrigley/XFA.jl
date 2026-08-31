struct XfaContextException <: Exception
    msg::String
end

struct XfaExecutionException <: Exception
    msg::String
end

@enum DependencyKind begin
    DepKind_Variable
    DepKind_Subvariable
    DepKind_Karabo
    DepKind_Group
    DepKind_GroupParameter
end

@kwdef struct Dependency
    kind::DependencyKind
    name::String

    # Karabo-specific
    topic::Union{String, Nothing} = nothing
    source::Union{String, Nothing} = nothing
    property::Union{String, Nothing} = nothing
    proxy::Union{String, Nothing} = nothing

    # Subvariable-specific
    parent::Union{String, Nothing} = nothing

    # Group-specific
    group_type::Union{DataType, Nothing} = nothing

    # GroupParameter-specific
    group_type_name::Union{String, Nothing} = nothing
    parameter::Union{String, Nothing} = nothing
end

function Base.show(io::IO, dep::Dependency)
    if dep.kind == DepKind_Karabo
        print(io, "karabo\"$(dep.name)\"")
    elseif dep.kind ∈ (DepKind_Variable, DepKind_Subvariable)
        print(io, Dependency, "(\"$(dep.name)\")")
    else
        print(io, Dependency, "(kind=$(dep.kind), name=$(dep.name))")
    end
end

Base.string(dep::Dependency) = dep.name

# Variable dependency
Dependency(name::String) = Dependency(kind=DepKind_Variable, name=name)

# Subvariable dependency
subvariable_dependency(parent::AbstractString, name::AbstractString) = Dependency(kind=DepKind_Subvariable, name="$parent.$name", parent=parent)

# Group dependency
group_dependency(type::DataType) = Dependency(kind=DepKind_Group, name="", group_type=type)
group_dependency(::Nothing, type::DataType) = group_dependency(type)
group_dependency(name::String, type::DataType) = Dependency(kind=DepKind_Group, name=name, group_type=type)

# GroupParameter dependency: used when a group variable's argument references
# one of its group's Parameter fields (e.g. `@Variable foo(::MyGroup, data ->
# MyGroup.data_param)`). At load time this is resolved to the actual dependency
# value held by the parameter.
function group_parameter_dependency(group_type::AbstractString, parameter::AbstractString)
    Dependency(kind=DepKind_GroupParameter,
               name="$group_type.$parameter",
               group_type_name=group_type,
               parameter=parameter)
end

@kwdef mutable struct Parameter{T}
    name::String = ""
    value::Union{T, Nothing} = nothing
    set_by_user::Bool = false

    update_handler::Union{Function, Nothing} = nothing
    initializer::Union{Function, Nothing} = nothing
    optional::Bool = false
end

@enum VariableKind VariableKind_Variable VariableKind_Group VariableKind_Input

# Wire-safe description of an available @Variable/@Group/@Input for the GUI's
# add-variable dialog (see `available_variables`). `dependencies` are the args to
# wire (empty for groups/inputs); `group_parameters` holds the default Parameter
# object for each Parameter-typed group field (handlers stripped for wire safety).
struct VariableSpec
    name::String
    kind::VariableKind
    origin::String                              # module-qualified ref, e.g. "XfaContext.scan"
    dependencies::OrderedDict{String, Dependency}
    subvariables::Vector{String}
    postprocessors::Vector{String}
    group_parameters::OrderedDict{Symbol, Parameter}
end

# Build a spec from a registered @Variable function.
function VariableSpec(func::Function)
    dependencies = OrderedDict{String, Dependency}()
    for (arg_name, dep) in variable_dependencies(func)
        if isnothing(arg_name) || !(dep isa Dependency)
            continue
        end
        # Strip group_type: it may name a module the client hasn't loaded.
        dependencies[arg_name] = isnothing(dep.group_type) ? dep : @set dep.group_type = nothing
    end

    VariableSpec(string(nameof(func)), VariableKind_Variable, origin_path(func),
                 dependencies, variable_subvariables(func),
                 [pp[1] for pp in variable_postprocessors(func)], OrderedDict{Symbol, Parameter}())
end

# Whether a @Group type backs an @Input, making it an input source not a plain group.
function is_input_group(T::DataType)
    for m in methods(input_dependencies)
        F = m.sig.parameters[2]
        if !isdefined(F, :instance)
            continue
        end
        deps = input_dependencies(F.instance)
        if !isempty(deps) && deps[1][2] isa Dependency && deps[1][2].group_type === T
            return true
        end
    end
    return false
end

# Build a spec from a registered @Group type, deriving its kind (Group/Input).
function VariableSpec(T::DataType)
    kind = is_input_group(T) ? VariableKind_Input : VariableKind_Group
    # Strip the update handlers/initializers so the spec stays wire-safe (they
    # may be closures or functions the client doesn't have). The defaults are
    # freshly built each call, so mutating them is safe.
    params = group_default_parameters(T)
    for p in values(params)
        p.update_handler = nothing
        p.initializer = nothing
    end
    VariableSpec(string(nameof(T)), kind, origin_path(T),
                 OrderedDict{String, Dependency}(), String[], String[], params)
end

abstract type AbstractPostprocessor end
function default_name end

# Rectangular region-of-interest, used as a Parameter value type. The GUI
# overlays a `RectROI`-valued Parameter on its variable's image plot when the
# variable declares it via `@display`.
struct RectROI
    corner_x::Float64
    corner_y::Float64
    width::Float64
    height::Float64
end

# Zero-initialized ROI; the GUI treats all-zero as uninitialized and picks a
# default extent based on the image being plotted.
RectROI() = RectROI(0.0, 0.0, 0.0, 0.0)

Base.isassigned(roi::RectROI) = !(roi.corner_x == 0 && roi.corner_y == 0 && roi.width == 0 && roi.height == 0)

function (r::RectROI)(image::AbstractArray)
    corner_x = round(Int, r.corner_x)
    corner_y = round(Int, r.corner_y)
    width = round(Int, r.width)
    height = round(Int, r.height)

    x_range = axes(image, 2)
    y_range = axes(image, 1)

    min_x = clamp(corner_x, x_range)
    max_x = clamp(min_x + width, x_range)
    min_y = clamp(corner_y, y_range)
    max_y = clamp(min_y + height, y_range)

    # Slice the first two dimensions with the ROI and keep all remaining
    # dimensions in full (e.g. a 3D stack yields a stack of cropped images).
    trailing = ntuple(_ -> Colon(), ndims(image) - 2)
    @view image[min_y:max_y, min_x:max_x, trailing...]
end

struct OptionalDims
    dims::Union{Vector{Int}, Vector{String}}
end
OptionalDims() = OptionalDims(Int[])

const slow_data_re = r"^(\S+?)\.([\w|\.]+)$"
const fast_data_re = r"^(\S+):(\S+)\[(\S+)\]$"
const topic_prefix_re = r"^(\w+)//(.+)$"

# Compute the string representation for a Karabo dependency
function karabo_dep_string(topic, source, property, proxy=nothing)
    device_str = if occursin(':', source)
        "$(source)[$(property)]"
    else
        "$(source).$(property)"
    end

    base = isnothing(topic) ? device_str : "$(topic)//$(device_str)"
    return isnothing(proxy) ? base : "$(base)@$(proxy)"
end

# Source string for a Karabo dep as expected by the TrainMatcher's `sources`
# field: `DEVICE:output` for fast/pipeline data, `DEVICE.property` for slow data.
function trainmatcher_dep_string(dep::Dependency)
    base = occursin(':', dep.source) ? dep.source : "$(dep.source).$(dep.property)"
    isnothing(dep.proxy) ? base : "$(base)@$(dep.proxy)"
end

# Karabo dependency constructors
karabo_dependency(source::AbstractString, property::AbstractString) = karabo_dependency(nothing, source, property, nothing)

function karabo_dependency(topic, source, property, proxy=nothing)
    name = karabo_dep_string(topic, source, property, proxy)
    Dependency(kind=DepKind_Karabo, name=name,
               topic=isnothing(topic) ? nothing : String(topic),
               source=String(source), property=String(property),
               proxy=isnothing(proxy) ? nothing : String(proxy))
end

function karabo_dependency(str::AbstractString)
    topic = nothing
    proxy = nothing

    proxy_idx = findlast('@', str)
    if !isnothing(proxy_idx)
        proxy = str[proxy_idx+1:end]
        str = str[1:proxy_idx-1]
    end

    m = match(topic_prefix_re, str)
    if !isnothing(m)
        topic = m.captures[1]
        str = m.captures[2]
    end

    m = match(slow_data_re, str)
    if !isnothing(m)
        return karabo_dependency(topic, m.captures[1], m.captures[2], proxy)
    end

    m = match(fast_data_re, str)
    if !isnothing(m)
        return karabo_dependency(topic, "$(m.captures[1]):$(m.captures[2])", m.captures[3], proxy)
    end

    throw(ArgumentError("'$(str)' is not a valid Karabo device property"))
end

"""
@karabo_str macro for creating Karabo device dependencies.

Example usage:
    karabo"MID_EXP_UPP/MOTOR/T5.actualPosition"
    karabo"SA2_XTD1_XGM/XGM/DOOCS:output[data.intensityTD]"
"""
macro karabo_str(str)
    Expr(:call, :karabo_dependency, esc(Base.Meta.parse("\"$(escape_string(str))\"")))
end


"""
Helper function to parse the arguments of a function.
"""
function _parse_function_args(args; is_input=false)
    dependencies = []
    group_type_name = nothing  # Set when first arg is a GroupDependency
    new_args = [postwalk(arg) do arg_expr
                    if @capture(arg_expr, arg_name_ -> value_)
                        # Strip quote nodes etc
                        value = MacroTools.unblock(value)

                        if @capture(value, head_.tail_)
                            value_str = "$value"
                            if !isnothing(group_type_name) && startswith(value_str, "$(group_type_name).")
                                # Reference to a member of the surrounding group's type:
                                # a Parameter field, another @Variable, or a subvariable
                                # of one. Resolved at group instantiation time.
                                path = chopprefix(value_str, "$(group_type_name).")
                                value = :(Context.group_parameter_dependency($("$group_type_name"), $path))
                            else
                                # Subvariable reference: e.g. var.subvar
                                value = :(Context.subvariable_dependency($("$head"), $("$tail")))
                            end
                        elseif !isnothing(group_type_name)
                            throw(ArgumentError("Dependencies of @Group @Variable's must refer to a group parameter (e.g. $(group_type_name).<parameter>) or a subvariable of another group variable, got: $(value)"))
                        elseif value isa Dependency || @capture(value, @karabo_str _)
                            # Keep as-is
                        elseif value isa Symbol
                            value = :(Context.Dependency($("$value")))
                        else
                            throw(ArgumentError("Unrecognized dependency expression: $(value)"))
                        end
                        push!(dependencies, :(($("$arg_name"), $value)))

                        # Replace the original argument expression with just
                        # the argument name.
                        return arg_name

                    elseif @capture(arg_expr, (::T_) | (arg_name_::T_))
                        if !is_input || (is_input && i == 1 && length(args) == 2)
                            # If the first argument has a type and no explicit
                            # dependency, then we assume it belongs to a group.
                            # A @Variable's typed args always land here.
                            arg_name_expr = isnothing(arg_name) ? :(nothing) : :($("$arg_name"))
                            group_type_name = "$T"
                            push!(dependencies, :(($arg_name_expr, Context.group_dependency($T))))
                        end
                        # Otherwise (only reachable for an @Input) it's a plain
                        # positional argument such as the output channel, supplied
                        # at call time and not tracked as a dependency.

                        return arg_expr
                    else
                        return arg_expr
                    end
                end
                for (i, arg) in enumerate(args)]

    return dependencies, new_args
end

# Helper functions for detecting variable references. A reference can be a
# module-qualified name (MyLib.func), a bare symbol (func), or a call on
# either of those with dependency overrides (MyLib.func(data -> ...)).
_is_qualified_name(expr) = expr isa Expr && expr.head == :.
_is_ref_name(expr) = expr isa Symbol || _is_qualified_name(expr)
_is_variable_ref(expr) = _is_ref_name(expr) || (expr isa Expr && expr.head == :call && _is_ref_name(expr.args[1]))

function _ref_basename(expr)
    if expr isa Expr && expr.head == :call
        return _ref_basename(expr.args[1])
    elseif _is_qualified_name(expr)
        return expr.args[2].value
    elseif expr isa Symbol
        return expr
    end
end

function _split_ref(expr)
    if expr isa Expr && expr.head == :call
        return expr.args[1], expr.args[2:end]
    else
        return expr, []
    end
end

# Generates code for a variable that references an existing variable from
# another module. Creates a wrapper function and registers trait methods
# that delegate to the original, with optional dependency overrides.
function _variable_reference(new_name, ref_expr, side_effects)
    orig_func_expr, override_args = _split_ref(ref_expr)

    # Build dependency registration code
    if !isempty(override_args)
        overrides, _ = _parse_function_args(override_args)
        overrides_expr = Expr(:vect, overrides...)
        deps_code = quote
            let orig_deps = Context.variable_dependencies($orig_func_expr),
                overrides = Dict{String, Any}($overrides_expr)
                Context.variable_dependencies(::typeof($new_name)) = [(name, get(overrides, name, dep)) for (name, dep) in orig_deps]
            end
        end
    else
        deps_code = :(Context.variable_dependencies(::typeof($new_name)) = Context.variable_dependencies($orig_func_expr))
    end

    # Build subvariable and postprocessor registration code, remapping
    # names if renamed. When names match the replace is a no-op.
    orig_name_str = string(_ref_basename(ref_expr))
    new_name_str = string(new_name)
    subvars_code = quote
        Context.variable_subvariables(::typeof($new_name)) = [replace(s, $orig_name_str => $new_name_str, count=1)
                                                              for s in Context.variable_subvariables($orig_func_expr)]
    end
    postprocessors_code = quote
        Context.variable_postprocessors(::typeof($new_name)) = [(replace(pp[1], $orig_name_str => $new_name_str, count=1), pp[2])
                                                                for pp in Context.variable_postprocessors($orig_func_expr)]
    end
    displays_code = quote
        Context.variable_displays(::typeof($new_name)) = Context.variable_displays($orig_func_expr)
    end

    return esc(quote
        function $new_name(args...)
            $orig_func_expr(args...)
        end

        if $side_effects
            Context.variable_origin(::typeof($new_name)) = $orig_func_expr
            $deps_code
            $subvars_code
            $postprocessors_code
            $displays_code
        end
    end)
end

function _variable(ctx_module, expr, side_effects)
    if !(expr isa Expr)
        throw(ArgumentError("Must pass an Expr to @Variable"))
    end

    # Handle the `->` shorthand form. The right-hand side can be:
    # - A Karabo dependency: @Variable cam -> karabo"camera.pixels"
    # - A variable reference (qualified or bare symbol), optionally with
    #   dependency overrides:
    #     @Variable my_norm -> MyLib.normalize
    #     @Variable my_norm -> normalize
    #     @Variable my_norm -> MyLib.normalize(data -> karabo"other.source")
    #   Bare symbols are only allowed here (not without ->), because without
    #   a rename the new wrapper function would conflict with the imported name.
    if @capture(expr, name_ -> value_)
        # Strip quote nodes etc
        value = MacroTools.unblock(value)

        if (value isa Dependency && value.kind == DepKind_Karabo) || (value isa Expr && @capture(value, @karabo_str _))
            function_expr = quote
                function $name(data -> $value)
                    return data
                end
            end

            # Recurse with the generated function
            return _variable(ctx_module, function_expr, side_effects)
        elseif _is_variable_ref(value)
            return _variable_reference(name, value, side_effects)
        else
            throw(ArgumentError("Unrecognized dependency: $(value)"))
        end

    elseif @capture(expr, function func_name_(args__) body_ end)
        # And now we handle explicit function declarations

        # Extract dependency information
        dependencies, new_args = _parse_function_args(args)

        # Identify the first typed arg, which (per _parse_function_args) is
        # treated as a group binding. Its name lets @display g.field resolve
        # to the right group type.
        group_arg_name = nothing
        group_type_name = nothing
        for arg in args
            if @capture(arg, (::T_) | (arg_name_::T_))
                group_arg_name = isnothing(arg_name) ? nothing : string(arg_name)
                group_type_name = string(T)
                break
            end
        end

        # Look through the body for @add_subvariable, @postprocess, and
        # @display calls to register them. Only toplevel calls are allowed.
        # Note that we capture the macro name and check it explicitly because
        # MacroTools doesn't handle the underscore in the name properly.
        subvariables = String[]
        postprocessors = []  # (name_expr, pp_expr) tuples
        postprocessor_indices = Int[]
        displays = String[]  # fully-qualified parameter names
        display_indices = Int[]
        for (i, body_expr) in enumerate(body.args)
            if @capture(body_expr, @macroname_(subvar_name_, _)) && macroname == Symbol("@add_subvariable")
                push!(subvariables, "$(func_name).$(subvar_name)")
            elseif @capture(body_expr, @macroname_(pp_name_, pp_expr_)) && macroname == Symbol("@postprocess")
                push!(postprocessors, (pp_name, pp_expr))
                push!(subvariables, "$(func_name).$(pp_name)")
                push!(postprocessor_indices, i)
            elseif @capture(body_expr, @macroname_(pp_expr_)) && macroname == Symbol("@postprocess")
                push!(postprocessors, (nothing, pp_expr))
                push!(postprocessor_indices, i)
            elseif @capture(body_expr, @macroname_(disp_expr_)) && macroname == Symbol("@display")
                push!(display_indices, i)
                if @capture(disp_expr, head_.tail_)
                    if isnothing(group_arg_name) || string(head) != group_arg_name
                        throw(ArgumentError("@display: '$(head)' is not the group argument of this @Variable"))
                    end
                    push!(displays, "$(group_type_name).$(tail)")
                elseif disp_expr isa Symbol
                    push!(displays, string(disp_expr))
                else
                    throw(ArgumentError("@display takes a parameter name or group.field, got: $(disp_expr)"))
                end
            end
        end

        postwalk(body) do body_expr
            if @capture(body_expr, @macroname_(subvar_name_, _)) && macroname == Symbol("@add_subvariable")
                if "$(func_name).$(subvar_name)" ∉ subvariables
                    throw(ArgumentError("Subvariable '$(func_name).$(subvar_name)' must be defined at the toplevel of the function"))
                end
            end

            body_expr
        end

        # Strip @postprocess and @display calls from the body (they're metadata, not runtime code)
        deleteat!(body.args, sort!(vcat(postprocessor_indices, display_indices)))

        # Build the postprocessors expression. Each entry evaluates the
        # constructor once, then resolves the name — either from the
        # explicit string or via default_name() for the one-arg form.
        func_name_str = "$func_name"
        postprocessors_expr = Expr(:vect,
            [:(let _pp = $(pp_expr)
                   _name = $(isnothing(name) ? :(Context.default_name(_pp)) : "$name")
                   ("$($(func_name_str)).$(_name)", _pp)
               end)
             for (name, pp_expr) in postprocessors]...)

        # Combine all the dependency expressions into a vector expr, which will
        # get interpolated/evaluated properly.
        dependencies_expr = Expr(:vect, dependencies...)
        subvariables_expr = Expr(:vect, subvariables...)
        displays_expr = Expr(:vect, displays...)
        new_function = quote
            function $func_name($(new_args...))
                $body
            end

            if $side_effects
                Context.variable_dependencies(::typeof($func_name)) = $dependencies_expr
                Context.variable_subvariables(::typeof($func_name)) = $subvariables_expr
                Context.variable_postprocessors(::typeof($func_name)) = $postprocessors_expr
                Context.variable_displays(::typeof($func_name)) = $displays_expr
            end
        end

        return esc(new_function)

    elseif _is_qualified_name(expr) || (expr isa Expr && expr.head == :call && _is_qualified_name(expr.args[1]))
        # Handle module-qualified references without rename:
        #   @Variable MyLib.func
        #   @Variable MyLib.func(data -> karabo"other.source")
        # Requires a qualified name (not a bare symbol) to avoid conflicting
        # with the imported binding.
        new_name = _ref_basename(expr)
        return _variable_reference(new_name, expr, side_effects)
    end

    throw(ArgumentError("Could not construct variable from expression: $(prettify(expr))"))
end

# Register a subvariable value in the current execution context. This is
# emitted by the @Variable macro when it encounters inner subvariable
# declarations, and should not be called directly.
macro add_subvariable(name, value)
    esc(quote
        let _val = $value
            _key = "$(Context.Meta.name[]).$($name)"
            Context.Meta.subvariables[][_key] = _val
            _val
        end
    end)
end

macro postprocess(args...)
    error("The @postprocess macro may only be used inside of a @Variable block.")
end

macro display(args...)
    error("The @display macro may only be used inside of a @Variable block.")
end

"""
Mark functions for execution in XFA.
"""
macro Variable(expr)
    _variable(__module__, expr, true)
end

function Base.:(==)(one::Parameter{T}, two::Parameter{T}) where T
    one.name == two.name && one.value == two.value && one.set_by_user == two.set_by_user
end

Parameter(name::String, value; optional=false) = Parameter(; name, value, optional)
Parameter(value; optional=false) = Parameter(; name="", value, optional)

# Used by @Group to allow passing raw values for Parameter fields
_wrap_param(val::Parameter) = val
_wrap_param(val) = Parameter(val)
# Merge a raw value into a default Parameter, preserving handlers
_wrap_param(val::Parameter, ::Parameter) = val
_wrap_param(val, default::Parameter) = Parameter(default.name, val, default.set_by_user,
                                                  default.update_handler, default.initializer,
                                                  default.optional)

function Parameter(f::Base.Callable, value; optional=false)
    if !isnothing(f) && !any(m -> m.nargs in (2, 3), methods(f))
        throw(ArgumentError("Parameter update handler must be either `nothing` or a callable that takes one argument (value) or two arguments (group, value)"))
    end

    Parameter("", value, false, f, f, optional)
end

function tryset(param::Parameter, value; force=false)
    if param.set_by_user && force
        param.set_by_user = false
    end

    if !param.set_by_user
        param.value = value
        requestor = Base.ScopedValues.get(Meta.name)
        if !isnothing(requestor)
            remote_do(set_parameter, 1, param.name, value, something(requestor))
        end
        return true
    else
        return false
    end
end

Base.getindex(param::Parameter) = param.value
Base.setindex!(param::Parameter, value) = param.value = value

Base.isassigned(param::Parameter) = !isnothing(param.value)

# Runs on proc 1 in response to a worker's `tryset`. Mirrors the new value
# into the coordinator's parameter dict and notifies the context's
# `on_parameter_changed` hook so the engine can broadcast it to clients.
function set_parameter(name::String, value, requestor::String)
    @info "Setting parameter '$(name)' to $(value) as requested by '$(requestor)'"

    if haskey(worker_state.parameters, name)
        pause_pipeline() do
            worker_state.parameters[name].value = value
        end
    end

    if !isnothing(current_ctx)
        current_ctx.on_parameter_changed(name, value)
    end
end

function _input(ctx_module, expr, side_effects)
    if !(expr isa Expr)
        throw(ArgumentError("Must pass an Expr to @Input"))
    end

    if @capture(expr, function name_(args__) body_ end)
        dependencies, new_args = _parse_function_args(args; is_input=true)
        if length(new_args) == 2
            if isempty(dependencies) || !@capture(dependencies[1], (_, Context.group_dependency(_)))
                throw(XfaContextException("The first argument of a two-argument @Input must be a @Group"))
            end
        elseif length(new_args) != 1
            throw(XfaContextException("@Input functions must accept 1-2 arguments, '$(name)' has $(length(args)) arguments"))
        end

        dependencies_expr = Expr(:vect, dependencies...)
        new_expr = quote
            function $name($(args...))
                $body
            end

            if $side_effects
                Context.input_dependencies(::typeof($name)) = $dependencies_expr
            end
        end

        return esc(new_expr)
    end

    throw(ArgumentError("Could not construct an input from expression: $(prettify(expr))"))
end

"""
Mark a function as an input (i.e a trigger).
"""
macro Input(expr)
    _input(__module__, expr, true)
end

struct Group
    type::DataType
    variables::Vector{Function}
end

function Base.:(==)(x::Group, y::Group)
    x.type === y.type && x.parameters == y.parameters && x.variables == y.variables
end

# Generate a struct definition and keyword constructor for @Group structs.
# Raw values passed for Parameter{T} fields are automatically wrapped, preserving
# any handlers from the default Parameter if one exists.
function _kwdef_group(name, struct_expr, fields)
    is_mutable = struct_expr.args[1]

    parsed = []
    extra_exprs = []
    for field in fields
        default = nothing
        if @capture(field, lhs_ = rhs_)
            default = rhs
            field = lhs
        end

        if @capture(field, fname_::ftype_)
            is_param = @capture(ftype, Parameter{_})
        elseif field isa Symbol
            fname = field
            is_param = false
        else
            push!(extra_exprs, field)
            continue
        end

        push!(parsed, (; name=fname, default, is_param, typed_field=field))
    end

    struct_fields = [f.typed_field for f in parsed]
    struct_def = Expr(:struct, is_mutable, name, Expr(:block, struct_fields..., extra_exprs...))

    kwargs = [isnothing(f.default) ? f.name : Expr(:kw, f.name, f.default)
              for f in parsed]
    wrap_stmts = [if !isnothing(f.default)
                      :($(f.name) = Context._wrap_param($(f.name), $(f.default)))
                  else
                      :($(f.name) = Context._wrap_param($(f.name)))
                  end
                  for f in parsed if f.is_param]
    field_names = [f.name for f in parsed]

    ctor = if isempty(parsed)
        nothing
    else
        :(function $name(; $(kwargs...))
            $(wrap_stmts...)
            $name($(field_names...))
        end)
    end

    # Default Parameter objects for the Parameter-typed fields, keyed by field
    # name. Fields with a default reuse it (via _wrap_param); fields without one
    # get an empty Parameter of the declared type. Used to build a VariableSpec.
    param_pairs = [:($(QuoteNode(f.name)) => let p = $(isnothing(f.default) ?
                         :($(f.typed_field.args[2])()) :
                         :(Context._wrap_param($(f.default))))
                         p.name = $(string(f.name))
                         p
                     end)
                   for f in parsed if f.is_param]
    param_defaults = :(Context.OrderedDict{Symbol, Context.Parameter}($(param_pairs...)))

    return struct_def, ctor, param_defaults
end

function _group(ctx_module, expr, side_effects)
    if !(expr isa Expr)
        throw(ArgumentError("Must pass an Expr to @Group"))
    end

    if expr.head == :macrocall && expr.args[1] == Symbol("@kwdef")
        throw(ArgumentError("@kwdef is not needed with @Group, keyword constructors are generated automatically"))
    end

    if @capture(expr, struct name_ fields__ end) || @capture(expr, mutable struct name_ fields__ end)
        struct_def, ctor, param_defaults = _kwdef_group(name, expr, fields)

        new_expr = quote
            $struct_def
            $ctor

            if $side_effects
                Context.group_fields(::Type{$name}) = []
                Context.group_default_parameters(::Type{$name}) = $param_defaults
            end

            $name
        end

        return esc(new_expr)
    end

    throw(ArgumentError("Could not construct a group from expression: $(prettify(expr))"))
end

"""
Mark a struct as a group of @Variable's.
"""
macro Group(expr)
    _group(__module__, expr, true)
end
