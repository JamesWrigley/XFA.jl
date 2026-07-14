module Protocol

export AbstractMessage, Ping, Shutdown,
    GetTrainmatchers, GetInputSources, LoadContext, ReviseCode,
    GetDeviceSchema, DeviceSchema,
    GetDeviceProperty, DeviceProperty,
    GetEngineDir, EngineDir,
    GetRoutingRules, SetRoutingRules, RoutingRules,
    GetRemapRules, RemapRules,
    SetVariableSubscriptions,
    ChangeParameter, Start, Stop,
    SetDebugMode, SetRemoteRepl,
    Pong, AvailableTrainmatchers,
    Started, Stopped, InputSources,
    ContextInfo, ParameterChanged, TrainData, RemoteReplState,
    PipelineStats, Ack, Envelope, MessageId, ExceptionMessage, client_send, server_send

import Serialization: serialize, deserialize

import HTTP: WebSockets

import XfaContext
using XfaContext: ContextState, VariableData, Parameter, SourceInfo
using ..XfaEngine: RoutingRule, RemapRule


# Max websocket frame size for client-engine traffic. The HTTP.jl default of
# 16 MiB is too small so we effectively disable the limit.
const MAX_FRAME_SIZE = typemax(Int)

abstract type AbstractMessage end

# Wraps a printed exception + stacktrace as a plain string so the client
# never has to deserialize Exception subtypes whose modules it may not have
# loaded.
struct ExceptionMessage
    text::String
end
ExceptionMessage(ex, bt) = ExceptionMessage(sprint(showerror, ex, bt))
ExceptionMessage(ex::Exception) = ExceptionMessage(sprint(showerror, ex))

# Messages that a client can send
struct Ping <: AbstractMessage end
struct Shutdown <: AbstractMessage end

struct GetDeviceSchema <: AbstractMessage
    topic::String
    name::String
end

struct GetDeviceProperty <: AbstractMessage
    topic::String
    device::String
    property::String
end

struct LoadContext <: AbstractMessage
    path::String
end

struct ReviseCode <: AbstractMessage end

struct ChangeParameter <: AbstractMessage
    parameter::Parameter
end

struct GetRoutingRules <: AbstractMessage end

struct GetRemapRules <: AbstractMessage end

struct SetRoutingRules <: AbstractMessage
    rules::Vector{RoutingRule}
end

# Tells the engine which array-valued variables this client wants forwarded.
# Scalar variables are always sent; everything else is suppressed unless its
# fully-qualified name (e.g. "var", "var.subvar") is a key in this dict.
# The value is the noise multiplier k for fixed-accuracy compression
# (tol = k * per-frame noise sigma); 0 means lossless and -1 means "use the
# engine default".
struct SetVariableSubscriptions <: AbstractMessage
    variables::Dict{String, Float64}
end

struct Start <: AbstractMessage end
struct Stop <: AbstractMessage end

struct SetDebugMode <: AbstractMessage
    enable::Bool
end

struct SetRemoteRepl <: AbstractMessage
    enable::Bool
end

struct GetTrainmatchers <: AbstractMessage end

struct GetInputSources <: AbstractMessage end

struct GetEngineDir <: AbstractMessage end

# Messages that the server can send
struct Pong <: AbstractMessage end

struct AvailableTrainmatchers <: AbstractMessage
    topic_trainmatchers::Dict{String, Vector{Tuple{String, Bool}}}
end

struct RoutingRules <: AbstractMessage
    rules::Vector{RoutingRule}
end

struct RemapRules <: AbstractMessage
    rules::Vector{RemapRule}
end

struct EngineDir <: AbstractMessage
    path::String
end

struct Started <: AbstractMessage end
struct Stopped <: AbstractMessage end

# The sources reported by each input of the loaded context, keyed by input name.
struct InputSources <: AbstractMessage
    input_sources::Union{Dict{String, Vector{SourceInfo}}, ExceptionMessage}
end

struct DeviceSchema <: AbstractMessage
    topic::String
    name::String
    schema::Dict{String, Dict}
end

struct DeviceProperty <: AbstractMessage
    topic::String
    device::String
    property::String
    value::Any
end

struct ContextInfo <: AbstractMessage
    info::Union{Dict, ExceptionMessage}
    is_running::Bool
    source::String
end

ContextInfo(ctx::ContextState, source::String) = ContextInfo(XfaContext.to_dict(ctx), ctx.is_running[], source)

struct ParameterChanged <: AbstractMessage
    parameter::Parameter
end

struct TrainData <: AbstractMessage
    variables::Vector{VariableData}
end

struct RemoteReplState <: AbstractMessage
    enabled::Bool
    port::Int
end

# Periodic pipeline metrics broadcast. `channel_stats` holds per-channel
# snapshots keyed by (producer, consumer); producer is either an external
# dependency name (e.g. "motor.pos") or a variable name, consumer is always
# the downstream variable name. `input_rates` is the smoothed Hz at which
# each input is pushing data, keyed by input name.
struct PipelineStats <: AbstractMessage
    channel_stats::Dict{Tuple{String, String}, XfaContext.ChannelStat}
    input_rates::Dict{String, Float64}
end

struct Ack <: AbstractMessage
    error::Union{ExceptionMessage, Nothing}
end
Ack() = Ack(nothing)

const MessageId = Int

struct Envelope
    id::MessageId
    reply_to::Union{MessageId, Nothing}
    msg::AbstractMessage
end

const _client_counter = Threads.Atomic{Int}(1)
const _server_counter = Threads.Atomic{Int}(-1)
next_client_id() = Threads.atomic_add!(_client_counter, 1)
next_server_id() = Threads.atomic_add!(_server_counter, -1)

function _send(ws::WebSockets.WebSocket, id::MessageId, msg::AbstractMessage;
               reply_to::Union{MessageId, Nothing}=nothing)
    envelope = Envelope(id, reply_to, msg)
    buffer = IOBuffer()
    serialize(buffer, envelope)
    WebSockets.send(ws, take!(buffer))
    return envelope.id
end

function client_send(ws::WebSockets.WebSocket, msg::AbstractMessage;
                     reply_to::Union{MessageId, Nothing}=nothing)
    _send(ws, next_client_id(), msg; reply_to)
end

function server_send(ws::WebSockets.WebSocket, msg::AbstractMessage;
                     reply_to::Union{MessageId, Nothing}=nothing)
    _send(ws, next_server_id(), msg; reply_to)
end

function receive(ws::WebSockets.WebSocket)::Envelope
    buffer = IOBuffer(WebSockets.receive(ws))
    return deserialize(buffer)
end

end
