import HTTP
import JSON3

# @sum_type WebProxyClientStatus :hidden begin
#     UNCONNECTED
#     CONNECTING
#     CONNECTED
#     ERROR
# end

const DEFAULT_WEBPROXY_ADDRESSES = Dict(
    "SA1" => "sa1-br-sys-con-gui1:8484",
    "FXE" => "fxe-rr-sys-con-gui1:8484",
    "SPB" => "spb-rr-sys-con-gui1:8484",

    "SA2" => "sa2-br-sys-con-gui1:8484",
    "MID" => "mid-rr-sys-con-gui1:8484",
    "HED" => "hed-rr-sys-con-gui1:8484",

    "SA3" => "sa3-br-sys-con-gui1:8484",
    "SCS" => "scs-rr-sys-con-gui1:8484",
    "SQS" => "sqs-rr-sys-con-gui1:8484",
    "SXP" => "sxp-rr-sys-con-gui1:8484"
)


const TimestampedResponse = @NamedTuple{timestamp::Float64, value::Dict}
mutable struct WebProxy
    address::String

    topology_cache::TimestampedResponse
    devices_cache::TimestampedResponse
end

function Base.show(io::IO, wp::WebProxy)
    print(io, WebProxy, "(\"$(wp.address)\")")
end

function WebProxy(address)
    if !startswith(address, "http://")
        address = "http://" * address
    end

    default_cache = (timestamp=-1.0, value=Dict())
    return WebProxy(address, default_cache, default_cache)
end

"""
    get_webproxy(device::AbstractString)

Get the webproxy for a device by extracting the topic from its name.
Device names are expected to start with the topic, e.g. `MID_FOO_BAR/...`
uses the `MID` topic webproxy.
"""
function get_webproxy(device::AbstractString)
    if isnothing(current_engine_state)
        error("Engine is not initialized, cannot get a WebProxy")
    elseif isempty(current_engine_state.webproxies)
        error("No WebProxy's are available to connect to")
    end

    for topic in keys(current_engine_state.webproxies)
        if startswith(device, topic)
            return current_engine_state.webproxies[topic]
        end
    end

    error("No webproxy found for device '$device'")
end

function get_webproxy(device::KaraboDevice)
    if isnothing(current_engine_state)
        error("Engine is not initialized, cannot get a WebProxy")
    end
    current_engine_state.webproxies[device.topic]
end

# Flatten the returned dict a bit by removing timestamp and tid entries for each
# node.
function strip_metadata!(x)
    for key in keys(x)
        if x[key] isa Dict && keys(x[key]) == Set(["value", "timestamp", "tid"])
            x[key] = x[key]["value"]
        end

        if x[key] isa Dict || x[key] isa Vector
            strip_metadata!(x[key])
        end
    end
end

function get_json(wp, path; timeout=5)
    res = HTTP.get(wp.address * path;
                   connect_timeout=timeout, readtimeout=timeout)
    return JSON3.read(res.body, Dict{String, Any})
end

function put_json(wp, path, body; timeout=5)
    res = HTTP.put(wp.address * path,
                   ["Content-Type" => "application/json"],
                   JSON3.write(body);
                   connect_timeout=timeout, readtimeout=timeout)
    return JSON3.read(res.body, Dict{String, Any})
end

function get_topology(wp; timeout=5, max_age=10)
    if time() - wp.topology_cache.timestamp > max_age
        value = get_json(wp, "/topology.json"; timeout)
        wp.topology_cache = (; timestamp=time(), value)
    end

    return wp.topology_cache.value
end

function get_devices(wp; timeout=5, max_age=10, classId=nothing)
    if time() - wp.devices_cache.timestamp > max_age
        value = get_json(wp, "/devices.json"; timeout)["devices"]
        wp.devices_cache = (; timestamp=time(), value)
    end

    devices = wp.devices_cache.value
    if !isnothing(classId)
        devices = filter(p -> p.second["classId"] == classId, devices)
    end

    return devices
end

function get_all_devices(webproxies; classId=nothing)
    devices = Dict{String, Dict{String, Any}}()
    for (topic, wp) in webproxies
        try
            devices[topic] = get_devices(wp; classId)
        catch ex
            @error "Error when getting devices for $(topic)" exception=(ex, catch_backtrace())
        end
    end

    return devices
end

# This takes in a dict from topic to webproxy
function get_all_trainmatchers(webproxies::Dict)
    trainmatchers = Dict{String, Vector{Tuple{String, Bool}}}()

    try
        for (topic, wp) in webproxies
            devices = get_devices(wp; classId="TrainMatcher")
            whitelisted = try
                get_property(wp, "karabo/WebProxy/device", "devices")
            catch ex
                @warn "Failed to query webproxy whitelist for $topic" exception=(ex, catch_backtrace())
                String[]
            end
            names = sort!(collect(keys(devices)))
            trainmatchers[topic] = [(name, name in whitelisted) for name in names]
        end
    catch ex
        if ex isa HTTP.ConnectError
            @warn "Couldn't get trainmatchers" exception=ex
        else
            rethrow()
        end
    end

    return trainmatchers
end

function get_config(device::KaraboDevice; timeout=5)
    wp = get_webproxy(device)
    config = get_json(wp, "/devices/$(device.name)/config.json"; timeout)
    strip_metadata!(config)
    return config
end

function get_property(wp, device, property; timeout=5)
    config = get_json(wp, "/devices/$(device).$(property)/config.json"; timeout)
    return config["value"]
end

function put_property(device::KaraboDevice, property, value; timeout=5)
    wp = get_webproxy(device)
    put_json(wp, "/devices/$(device.name).$(property)/config.json", value; timeout)
end

function get_schema(device::KaraboDevice; timeout=5)
    wp = get_webproxy(device)
    return get_json(wp, "/devices/$(device.name)/schema.json"; timeout)
end

function call_slot(wp, device, slot, params=HTTP.nobody; timeout=5)
    url = wp.address * "/devices/$(device)/slot/$(slot).json"
    body = params isa Dict ? JSON3.write(params) : params
    res = HTTP.put(url, nothing, body; connect_timeout=timeout, readtimeout=timeout)
    return JSON3.read(res.body, Dict{String, Any})
end

function get_trainmatcher_address(address, device::String; index=1)
    config = get_config(address, "/devices/$(device)/config.json")

    config["zmqOutputs"]["value"][index]["address"]["value"]
end
