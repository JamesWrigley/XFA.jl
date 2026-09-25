# Compile step for Vega-Lite plot specs. A spec Dict is validated against the
# supported subset and turned into a PlotSpec, so nothing downstream touches the
# Dict. Anything outside the subset throws a SpecError for the plot window to
# show.

struct SpecError <: Exception
    msg::String
end

# The Vega schemes and their ImPlot colormaps. Turbo's is only registered at
# runtime (turbo_colormap()), see scheme_colormap.
const COLOR_SCHEMES = ("viridis" => ImPlot.ImPlotColormap_Viridis, "plasma" => ImPlot.ImPlotColormap_Plasma,
                       "turbo" => nothing, "greys" => ImPlot.ImPlotColormap_Greys,
                       "spectral" => ImPlot.ImPlotColormap_Spectral, "rdbu" => ImPlot.ImPlotColormap_RdBu,
                       "brbg" => ImPlot.ImPlotColormap_BrBG, "piyg" => ImPlot.ImPlotColormap_PiYG,
                       "set1" => ImPlot.ImPlotColormap_Dark, "pastel1" => ImPlot.ImPlotColormap_Pastel,
                       "paired" => ImPlot.ImPlotColormap_Paired)

function check_keys(dict, allowed, path)
    for key in keys(dict)
        if !(key in allowed)
            throw(SpecError("$(path): unsupported property \"$(key)\""))
        end
    end
end

# dict[key] checked to be a T, or `default` if the key is absent.
function property(dict, key, T, path, default)
    if haskey(dict, key)
        value = dict[key]
        if !(value isa T)
            throw(SpecError("$(path).$(key): expected $(T), got $(typeof(value))"))
        end
        value
    else
        default
    end
end

# Required variant.
function property(dict, key, T, path)
    if !haskey(dict, key)
        throw(SpecError("$(path): missing \"$(key)\""))
    end
    property(dict, key, T, path, nothing)
end

# The (path, object) pairs of an optional array-of-objects property.
function objects(dict, key, path)
    elements = property(dict, key, AbstractVector, path, ())
    map(enumerate(elements)) do (i, element)
        element_path = "$(path).$(key)[$(i)]"
        if !(element isa AbstractDict)
            throw(SpecError("$(element_path): expected an AbstractDict, got $(typeof(element))"))
        end
        element_path, element
    end
end

# The value paired with `name` in `options`.
function choice(options, name, path)
    for (option, value) in options
        if option == name
            return value
        end
    end
    throw(SpecError("$(path): unsupported value \"$(name)\", expected one of: $(join(first.(options), ", "))"))
end

# A guide title: `default` when absent, hidden when null.
function guide_title(dict, path, default)
    title = property(dict, "title", Union{Nothing, AbstractString}, path, default)
    isnothing(title) ? "" : String(title)
end

# `positional` is true for x/y, false for color. An untyped field follows
# Vega-Lite's inference: quantitative with `bin` or a scale type, else nominal.
function compile_channel(def, path, positional::Bool)
    check_keys(def, positional ? ("field", "type", "title", "scale", "axis", "bin") :
                                 ("field", "type", "title", "scale"), path)
    field = String(property(def, "field", AbstractString, path))

    scale = property(def, "scale", AbstractDict, path, Dict{String, Any}())
    check_keys(scale, positional ? ("type",) : ("type", "scheme"), "$(path).scale")
    scale_type = property(scale, "type", AbstractString, "$(path).scale", nothing)
    log = if isnothing(scale_type)
        false
    else
        choice(("linear" => false, "log" => true), scale_type, "$(path).scale.type")
    end
    scheme = property(scale, "scheme", AbstractString, "$(path).scale", nothing)
    if !isnothing(scheme)
        choice(COLOR_SCHEMES, scheme, "$(path).scale.scheme")
    end

    binned = false
    if haskey(def, "bin")
        bin = property(def, "bin", AbstractDict, path)
        check_keys(bin, ("binned",), "$(path).bin")
        binned = property(bin, "binned", Bool, "$(path).bin")
        if !binned
            throw(SpecError("$(path).bin: only {\"binned\": true} is supported"))
        end
    end

    type = if haskey(def, "type")
        choice(("quantitative" => FieldType_Quantitative, "nominal" => FieldType_Nominal,
                "ordinal" => FieldType_Ordinal),
               property(def, "type", AbstractString, path), "$(path).type")
    elseif binned || !isnothing(scale_type)
        FieldType_Quantitative
    else
        FieldType_Nominal
    end

    title = guide_title(def, path, field)
    if haskey(def, "axis")
        axis = property(def, "axis", AbstractDict, path)
        check_keys(axis, ("title",), "$(path).axis")
        title = guide_title(axis, "$(path).axis", title)
    end

    ChannelDef(field, type, title, log, isnothing(scheme) ? nothing : String(scheme), binned)
end

# The mark and its opacity.
function compile_mark(unit, path)
    mark = property(unit, "mark", Union{AbstractString, AbstractDict}, path)
    opacity = 1.0
    if mark isa AbstractDict
        check_keys(mark, ("type", "opacity"), "$(path).mark")
        opacity = Float64(property(mark, "opacity", Real, "$(path).mark", 1.0))
        mark = property(mark, "type", AbstractString, "$(path).mark")
    end
    choice(("line" => Mark_Line, "point" => Mark_Point, "bar" => Mark_Bar, "rect" => Mark_Rect),
           mark, "$(path).mark"), opacity
end

# Up to two lookups are supported, sharing a key of `trainId` or `index` on both
# sides and each pulling one field.
function compile_lookups(unit, path)
    transforms = objects(unit, "transform", path)
    if length(transforms) > 2
        throw(SpecError("$(path).transform: at most two lookup transforms are supported"))
    end
    lookups = LookupTransform[compile_lookup(tpath, transform) for (tpath, transform) in transforms]
    if length(lookups) == 2 && lookups[1].key != lookups[2].key
        throw(SpecError("$(path).transform: the lookups must share a key"))
    end
    lookups
end

function compile_lookup(tpath, transform)
    if !haskey(transform, "lookup")
        throw(SpecError("$(tpath): only lookup transforms are supported"))
    end
    check_keys(transform, ("lookup", "from", "as"), tpath)
    key_name = property(transform, "lookup", AbstractString, tpath)
    key = choice(("trainId" => LookupKey_TrainId, "index" => LookupKey_Index), key_name, "$(tpath).lookup")

    from = property(transform, "from", AbstractDict, tpath)
    check_keys(from, ("data", "key", "fields"), "$(tpath).from")
    if property(from, "key", AbstractString, "$(tpath).from") != key_name
        throw(SpecError("$(tpath).from.key: must be the same as lookup (\"$(key_name)\")"))
    end
    from_data = property(from, "data", AbstractDict, "$(tpath).from")
    check_keys(from_data, ("name",), "$(tpath).from.data")
    dataset = property(from_data, "name", AbstractString, "$(tpath).from.data")

    fields = property(from, "fields", AbstractVector, "$(tpath).from")
    as = property(transform, "as", Any, tpath)
    if as isa AbstractVector && length(as) == 1
        as = as[1]
    end
    if length(fields) != 1 || !(fields[1] isa AbstractString) || !(as isa AbstractString)
        throw(SpecError("$(tpath): exactly one field must be pulled, with one \"as\" name"))
    end

    LookupTransform(key, String(dataset), String(fields[1]), String(as))
end

# Validates the channels against what each mark can draw: images are a rect
# with colour on `value`, everything else needs quantitative axes.
function compile_layer(unit, path)
    data = property(unit, "data", AbstractDict, path)
    check_keys(data, ("name",), "$(path).data")
    name = property(data, "name", AbstractString, "$(path).data")
    mark, opacity = compile_mark(unit, path)
    lookups = compile_lookups(unit, path)

    epath = "$(path).encoding"
    encoding = property(unit, "encoding", AbstractDict, path)
    check_keys(encoding, ("x", "y", "color"), epath)
    x = compile_channel(property(encoding, "x", AbstractDict, epath), "$(epath).x", true)
    y = compile_channel(property(encoding, "y", AbstractDict, epath), "$(epath).y", true)
    color = if haskey(encoding, "color")
        compile_channel(property(encoding, "color", AbstractDict, epath), "$(epath).color", false)
    else
        nothing
    end

    # One lookup pairs the other variable's values with the layer's own, two
    # place the layer's values at the points they give, coloured by value.
    fields = (x.field, y.field)
    if any(lookup -> lookup.field != "value" || lookup.as == "value", lookups)
        throw(SpecError("$(path): a lookup must pull \"value\" under another name"))
    elseif length(lookups) == 1
        as = lookups[1].as
        if mark == Mark_Rect || (fields != (as, "value") && fields != ("value", as))
            throw(SpecError("$(path): a lookup must plot against the layer's own \"value\" on x/y"))
        end
    elseif length(lookups) == 2
        as = (lookups[1].as, lookups[2].as)
        if mark != Mark_Point || as[1] == as[2] || (fields != as && fields != reverse(as)) ||
           isnothing(color) || color.field != "value" || color.type != FieldType_Quantitative
            throw(SpecError("$(path): two lookups must be points with x/y on the pulled fields, " *
                            "and a quantitative color on the layer's own \"value\""))
        end
    end

    if mark == Mark_Rect
        for (channel, axis) in ((x, "x"), (y, "y"))
            if channel.type == FieldType_Nominal
                throw(SpecError("$(epath).$(axis): a rect needs a quantitative or ordinal axis, add a \"type\""))
            end
        end
        if isnothing(color) || color.field != "value" || color.type != FieldType_Quantitative
            throw(SpecError("$(epath).color: a rect needs a quantitative color on the \"value\" field"))
        end
    else
        for (channel, axis) in ((x, "x"), (y, "y"))
            if channel.type != FieldType_Quantitative
                throw(SpecError("$(epath).$(axis): only quantitative axes are supported, add \"type\": \"quantitative\""))
            end
        end
        if !isnothing(color) && color.log && length(lookups) != 2
            throw(SpecError("$(epath).color.scale.type: a log color scale is only supported on a rect " *
                            "or points coloured by value"))
        end
    end

    LayerSpec(String(name), mark, opacity, x, y, color, lookups)
end

# The (lo, hi) of value[encoding].
function interval(value, encoding, path)
    range = property(value, encoding, AbstractVector, path)
    if length(range) != 2 || !all(v -> v isa Real, range)
        throw(SpecError("$(path).$(encoding): expected two numbers"))
    end
    minmax(Float64(range[1]), Float64(range[2]))
end

# Append the ROI params of `dict` (the spec or one of its layers) to `rois`.
function compile_params!(rois, dict, path)
    for (ppath, param) in objects(dict, "params", path)
        check_keys(param, ("name", "select", "value"), ppath)
        name = property(param, "name", AbstractString, ppath)

        select = property(param, "select", Union{AbstractString, AbstractDict}, ppath)
        encodings = ["x", "y"]
        if select isa AbstractDict
            check_keys(select, ("type", "encodings"), "$(ppath).select")
            encodings = property(select, "encodings", AbstractVector, "$(ppath).select", encodings)
            select = property(select, "type", AbstractString, "$(ppath).select")
        end
        if select != "interval"
            throw(SpecError("$(ppath).select: only interval selections are supported"))
        end
        if isempty(encodings) || !allunique(encodings) || !all(in(("x", "y")), encodings)
            throw(SpecError("$(ppath).select.encodings: expected \"x\", \"y\" or both"))
        end

        value = property(param, "value", AbstractDict, ppath, nothing)
        if !isnothing(value)
            check_keys(value, encodings, "$(ppath).value")
        end
        roi = if length(encodings) == 2
            if isnothing(value)
                RectROI()
            else
                x_lo, x_hi = interval(value, "x", "$(ppath).value")
                y_lo, y_hi = interval(value, "y", "$(ppath).value")
                RectROI(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo)
            end
        else
            axis = Symbol(encodings[1])
            if isnothing(value)
                LinearROI(; axis)
            else
                lo, hi = interval(value, encodings[1], "$(ppath).value")
                LinearROI(lo, hi - lo; axis)
            end
        end
        push!(rois, RoiParam(String(name), roi))
    end
end

function compile_model(model, path)
    check_keys(model, ("function", "params", "title"), path)
    func = choice(("gaussian" => ModelFunction_Gaussian,),
                  property(model, "function", AbstractString, path), "$(path).function")
    params = property(model, "params", AbstractString, path)
    title = property(model, "title", AbstractString, path, nothing)
    ModelOverlay(func, String(params), isnothing(title) ? nothing : String(title))
end

const UNIT_KEYS = ("data", "mark", "encoding", "transform", "params")

# The title defaults to `name`.
function compile_spec(name, spec::AbstractDict)
    top_keys = ("\$schema", "description", "title", "usermeta", "params")
    rois = RoiParam[]
    compile_params!(rois, spec, "spec")
    layers = if haskey(spec, "layer")
        check_keys(spec, (top_keys..., "layer"), "spec")
        units = objects(spec, "layer", "spec")
        if isempty(units)
            throw(SpecError("spec.layer: expected at least one layer"))
        end
        map(units) do (path, unit)
            check_keys(unit, UNIT_KEYS, path)
            compile_params!(rois, unit, path)
            compile_layer(unit, path)
        end
    else
        check_keys(spec, (top_keys..., UNIT_KEYS...), "spec")
        [compile_layer(spec, "spec")]
    end

    # Other tools may keep their own entries in usermeta, only `xfa` is ours
    usermeta = property(spec, "usermeta", AbstractDict, "spec", Dict{String, Any}())
    xfa_path = "spec.usermeta.xfa"
    xfa = property(usermeta, "xfa", AbstractDict, "spec.usermeta", Dict{String, Any}())
    check_keys(xfa, ("fixed_aspect", "models"), xfa_path)
    models = ModelOverlay[compile_model(model, path) for (path, model) in objects(xfa, "models", xfa_path)]

    PlotSpec(name, layers; title = property(spec, "title", AbstractString, "spec", name),
             rois, models, fixed_aspect = property(xfa, "fixed_aspect", Bool, xfa_path, true))
end

# Every variable a spec draws from, which its view subscribes to.
function datasets(spec::PlotSpec)
    names = Set{String}()
    for layer in spec.layers
        push!(names, layer.data)
        for lookup in layer.lookups
            push!(names, lookup.dataset)
        end
    end
    for model in spec.models
        push!(names, model.params)
    end
    names
end
