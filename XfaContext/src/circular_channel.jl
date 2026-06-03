# Bounded AbstractChannel whose `put!` never blocks: when the buffer is full,
# the oldest item is overwritten and `drop_count` is incremented. Used
# inter-variable data flow where a slow consumer should not stall upstream
# producers.
mutable struct CircularChannel{T} <: AbstractChannel{T}
    const buffer::CircularBuffer{T}
    const lock::ReentrantLock
    const cond_take::Threads.Condition
    @atomic drop_count::Int
    @atomic size::Int
    @atomic is_open::Bool
end

function CircularChannel{T}(capacity::Integer) where T
    l = ReentrantLock()
    CircularChannel{T}(CircularBuffer{T}(capacity), l, Threads.Condition(l), 0, 0, true)
end

drop_count(c::CircularChannel) = @atomic c.drop_count

# `size` mirrors `length(buffer)` but is maintained as an atomic so that
# `length()` can be queried lock-free (e.g. from telemetry running outside
# the producer/consumer tasks). put!/take! already hold the lock, so the
# atomic update is essentially free there.
Base.size(c::CircularChannel) = @atomic c.size
DataStructures.capacity(c::CircularChannel) = DataStructures.capacity(c.buffer)

function Base.put!(c::CircularChannel{T}, v) where T
    item = convert(T, v)
    @lock c.lock begin
        if !(@atomic c.is_open)
            throw(InvalidStateException("CircularChannel is closed.", :closed))
        end
        if isfull(c.buffer)
            @atomic c.drop_count += 1
        else
            @atomic c.size += 1
        end
        push!(c.buffer, item)
        notify(c.cond_take, nothing; all=false)
    end
    return v
end

function Base.take!(c::CircularChannel)
    @lock c.lock begin
        while isempty(c.buffer)
            if !(@atomic c.is_open)
                throw(InvalidStateException("CircularChannel is closed.", :closed))
            end
            wait(c.cond_take)
        end
        @atomic c.size -= 1
        return popfirst!(c.buffer)
    end
end

function Base.wait(c::CircularChannel)
    @lock c.lock begin
        while isempty(c.buffer)
            if !(@atomic c.is_open)
                throw(InvalidStateException("CircularChannel is closed.", :closed))
            end
            wait(c.cond_take)
        end
    end
    return nothing
end

Base.isready(c::CircularChannel) = @lock c.lock !isempty(c.buffer)
Base.isopen(c::CircularChannel) = @atomic c.is_open

function Base.close(c::CircularChannel)
    @lock c.lock begin
        @atomic c.is_open = false
        notify(c.cond_take, nothing; all=true)
    end
    return nothing
end
