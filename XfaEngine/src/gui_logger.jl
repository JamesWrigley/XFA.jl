# Set while broadcasting logs so that failures to send them aren't fed back
# into the GuiLogger.
const in_log_broadcast = ScopedValue(false)

# Collects log records for the clients. Records are tagged with the variable
# they were logged under (`Meta.name`), and consecutive repeats are merged
# until the next broadcast takes them.
@kwdef mutable struct GuiLogger <: AbstractLogger
    lock::ReentrantLock = ReentrantLock()
    min_level::LogLevel = Info
    pending::Vector{Protocol.LogRecord} = Protocol.LogRecord[]
    # Replayed to newly connected clients
    history::CircularBuffer{Protocol.LogRecord} = CircularBuffer{Protocol.LogRecord}(100)
end

Base.CoreLogging.min_enabled_level(logger::GuiLogger) = logger.min_level
Base.CoreLogging.shouldlog(::GuiLogger, args...) = !in_log_broadcast[]
Base.CoreLogging.catch_exceptions(::GuiLogger) = true

function Base.CoreLogging.handle_message(logger::GuiLogger, level, message, _module, group, id, file, line; kwargs...)
    variable = something(Base.ScopedValues.get(Context.Meta.name), "")
    tid = something(Base.ScopedValues.get(Context.Meta.tid), -1)
    message = string(message)
    key = hash((variable, id, message, kwargs_key(kwargs)))

    # The details are only formatted for the first of a set of repeats, and
    # outside the lock since formatting a backtrace is slow.
    merged = @lock logger.lock merge_pending!(logger, key, tid)
    if !merged
        details = format_details(kwargs)
        record = Protocol.LogRecord(; level, message, details, variable, tid, key,
                                    file=string(something(file, "")), line=Int(something(line, -1)))

        @lock logger.lock if !merge_pending!(logger, key, tid)
            push!(logger.pending, record)
        end
    end
end

# Must be called with the lock held. Returns whether the last pending record
# has the same key, in which case its count was incremented.
function merge_pending!(logger::GuiLogger, key, tid)
    if isempty(logger.pending)
        false
    else
        record = logger.pending[end]
        if record.key == key
            logger.pending[end] = setproperties(record; timestamp=time(), tid, count=record.count + 1)
            true
        else
            false
        end
    end
end

# Identifies the logged values so that records with different values, or the
# same exception thrown from elsewhere, aren't merged. Exceptions are compared
# by their text since most don't define equality, and the raw backtrace
# pointers are cheap to hash compared to formatting them.
function kwargs_key(kwargs)
    keys = Tuple{Symbol, Any}[]
    for (key, value) in kwargs
        if key == :maxlog
            continue
        end

        value_key = if key != :exception
            value
        elseif value isa Tuple{Any, Any}
            (sprint(showerror, value[1]), value[2])
        elseif value isa Base.ExceptionStack
            [(sprint(showerror, e.exception), e.backtrace) for e in value]
        else
            sprint(showerror, value)
        end
        push!(keys, (key, value_key))
    end

    keys
end

# Takes the records pending since the last call, adding them to the history
function take_pending!(logger::GuiLogger)
    @lock logger.lock begin
        records = logger.pending
        logger.pending = Protocol.LogRecord[]
        for record in records
            history = logger.history
            if !isempty(history) && history[end].key == record.key
                history[end] = setproperties(record; count=record.count + history[end].count)
            else
                push!(history, record)
            end
        end

        records
    end
end

log_history(logger::GuiLogger) = @lock logger.lock collect(logger.history)

function format_details(kwargs)
    io = IOBuffer()
    ctx = IOContext(io, :limit => true)
    for (key, value) in kwargs
        if key == :maxlog
            continue
        end

        if key == :exception
            if value isa Tuple{Any, Any}
                showerror(ctx, value[1], value[2])
            elseif value isa Exception
                showerror(ctx, value)
            else
                show(ctx, MIME"text/plain"(), value)
            end
        else
            print(ctx, key, " = ")
            show(ctx, MIME"text/plain"(), value)
        end
        println(ctx)
    end

    return String(rstrip(String(take!(io))))
end
