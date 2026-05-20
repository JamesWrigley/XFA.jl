# A rule for mapping (topic, source) to a specific input (trainmatcher device).
# Both `topic` and `source` are fnmatch-style globs; "*" matches anything.
struct RoutingRule
    topic::String
    source::String
    input::String
end

engine_settings_path() = abspath("engine.toml")

# Returns the rules from engine.toml, or nothing if the file or its
# `routing_rules` section is absent — lets the caller seed defaults on
# first run. An explicitly-empty section yields an empty vector.
function load_routing_rules()
    path = engine_settings_path()
    if !isfile(path)
        return nothing
    end

    settings = TOML.parsefile(path)
    if !haskey(settings, "routing_rules")
        return nothing
    end
    [RoutingRule(r["topic"], r["source"], r["input"]) for r in settings["routing_rules"]]
end

function write_routing_rules(rules::AbstractVector{RoutingRule})
    path = engine_settings_path()
    settings = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    settings["routing_rules"] = [Dict("topic" => r.topic,
                                      "source" => r.source,
                                      "input" => r.input) for r in rules]
    open(path, "w") do io
        TOML.print(io, settings; sorted=true)
    end
end

# Returns the input (trainmatcher device name) from the first rule whose topic
# and source globs both match, or nothing if no rule matches.
function match_rule(rules::AbstractVector{RoutingRule}, topic::AbstractString, source::AbstractString)
    for rule in rules
        if occursin(Glob.FilenameMatch(rule.topic), topic) &&
           occursin(Glob.FilenameMatch(rule.source), source)
            return rule.input
        end
    end
    return nothing
end

@enum RemapKind begin
    RemapKind_Simple
    RemapKind_Proxy
end

# A rule for rewriting a Karabo source string the user entered in the GUI.
# `source` and `device_class` are regexes (empty = match anything, AND'd
# together). `replacement` is a substitution string whose meaning depends on
# `kind`; the client owns interpretation. The struct just carries the data.
struct RemapRule
    kind::RemapKind
    source::String
    device_class::String
    replacement::String

    function RemapRule(kind, source, device_class, replacement)
        if isempty(source) && isempty(device_class)
            throw(ArgumentError("RemapRule must have at least one of `source` or `device_class` set"))
        end
        new(kind, source, device_class, replacement)
    end
end

# Hardcoded remap rules shipped with XFA. The combined effective list is
# `user_rules ++ BUILTIN_REMAP_RULES` so user rules always win.
const BUILTIN_REMAP_RULES = RemapRule[
    RemapRule(RemapKind_Simple,
              raw"^(.+):output\[data\.image\.pixels\]$",
              "AravisBaslerCamera",
              raw"\1:output[data.image.data]"),
    RemapRule(RemapKind_Proxy,
              raw"^.+:dataOutput\[.+\]$",
              "JungfrauCorrection",
              ""),
    RemapRule(RemapKind_Proxy,
              raw"^.+:dataOutput\[.+\]$",
              "Epix100Correction",
              "")
]

# Returns the effective list of remap rules: any user-defined rules from
# engine.toml followed by BUILTIN_REMAP_RULES (so user rules apply first).
function load_remap_rules()
    path = engine_settings_path()
    user = if !isfile(path)
        RemapRule[]
    else
        settings = TOML.parsefile(path)
        if !haskey(settings, "remap_rules")
            RemapRule[]
        else
            [RemapRule(RemapKind(r["kind"]), r["source"], r["device_class"], r["replacement"])
             for r in settings["remap_rules"]]
        end
    end
    vcat(user, BUILTIN_REMAP_RULES)
end

function write_remap_rules(rules::AbstractVector{RemapRule})
    path = engine_settings_path()
    settings = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    settings["remap_rules"] = [Dict("kind" => Int(r.kind),
                                    "source" => r.source,
                                    "device_class" => r.device_class,
                                    "replacement" => r.replacement) for r in rules]
    open(path, "w") do io
        TOML.print(io, settings; sorted=true)
    end
end
