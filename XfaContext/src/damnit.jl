# Something during a run that makes its saved data inconsistent
struct RunEvent
    tid::Int
    outputs::Vector{String}
    message::String
end

# The recorded trains of a saved output, with each train's sample flattened
# into `data` one after another. With SaveMode_Last only the latest sample is
# kept. The metadata is the latest sample's.
@kwdef mutable struct SaveBuffer{T}
    const mode::SaveMode
    const is_array::Bool
    const sample_size::Tuple{Vararg{Int}}
    const tids::Vector{Int} = Int[]
    const data::Vector{T}

    # The output's axes and labels
    meta::VariableData{Nothing} = VariableData(; data=nothing)
    # A DimArray's dims, refdims, name and metadata
    dims::Maybe{Tuple} = nothing
    refdims::Tuple = ()
    name::Any = DD.NoName()
    metadata::Any = DD.NoMetadata()
end

function SaveBuffer(mode::SaveMode, sample)
    is_array = sample isa AbstractArray
    T = is_array ? eltype(sample) : typeof(sample)
    sample_size = is_array ? size(sample) : ()
    # Reserve room for a few hundred trains up front so the vector doesn't
    # have to grow (reallocate and copy) often. It still grows by itself past
    # that.
    capacity = mode == SaveMode_PerTrain ? 300 : 1
    data = sizehint!(T[], capacity * prod(sample_size))
    SaveBuffer{T}(; mode, is_array, sample_size, data)
end

# Whether a sample can be stored with the buffer's earlier ones
function fits(buffer::SaveBuffer{T}, sample) where {T}
    if sample isa AbstractArray
        buffer.is_array && eltype(sample) == T && size(sample) == buffer.sample_size
    else
        !buffer.is_array && sample isa T
    end
end

# Store a train's sample. It has to be copied, since an output is only valid
# until its train has cleared the pipeline (e.g. MovingAvg returns its own
# buffer, and inputs may be views into re-used buffers).
function store!(buffer::SaveBuffer, out::VariableData)
    sample = out.data
    if buffer.mode == SaveMode_Last
        # Drop the previous sample, keeping the vector's memory for the next
        empty!(buffer.tids)
        empty!(buffer.data)
    end

    push!(buffer.tids, out.tid)
    if sample isa AbstractArray
        # This is the copy: append! grows `data` (reallocating when its
        # capacity runs out) and copies the sample's elements onto its end.
        append!(buffer.data, sample)
    else
        push!(buffer.data, sample)
    end

    buffer.meta = VariableData(; data=nothing, x_axis=out.x_axis, y_axis=out.y_axis,
                               xlabel=out.xlabel, ylabel=out.ylabel, unit=out.unit)
    if sample isa DD.AbstractDimArray
        buffer.dims = DD.dims(sample)
        buffer.refdims = DD.refdims(sample)
        buffer.name = DD.name(sample)
        buffer.metadata = DD.metadata(sample)
    else
        buffer.dims = nothing
    end
end

# A run being recorded, from `begin_run` until it's finalised
@kwdef mutable struct RunSession
    const proposal::Int
    const run::Int
    const first_tid::Int
    last_tid::Maybe{Int} = nothing
    const buffers::Dict{String, SaveBuffer} = Dict{String, SaveBuffer}()
    const events::Vector{RunEvent} = RunEvent[]
end

# The newest train recorded so far
latest_tid(session::RunSession) = maximum(b -> maximum(b.tids; init=session.first_tid), values(session.buffers); init=session.first_tid)

# A snapshot of the recorded run's bounds, which variables read every train to
# set Meta.proposal/Meta.run_number without taking the session lock.
struct RunBounds
    proposal::Int
    run::Int
    first_tid::Int
    last_tid::Maybe{Int}
end

RunBounds(session::RunSession) = RunBounds(session.proposal, session.run, session.first_tid, session.last_tid)

in_run(x::Union{RunSession, RunBounds}, tid) = tid >= x.first_tid && (isnothing(x.last_tid) || tid <= x.last_tid)

# The run being recorded, for showing its progress
struct RunStatus
    proposal::Int
    run::Int
    ntrains::Int
    nbytes::Int
    saving::Bool
end

# Read from the context's atomics, so it doesn't take the session lock. A run
# being saved is shown over one being recorded.
function run_status(ctx)
    saving = ctx.saving_run[]
    bounds = ctx.run_bounds[]
    if !isnothing(saving)
        saving
    elseif isnothing(bounds)
        nothing
    else
        RunStatus(bounds.proposal, bounds.run, ctx.run_ntrains[], ctx.run_nbytes[], false)
    end
end

function parse_save_mode(mode::Symbol)
    if mode == :per_train
        SaveMode_PerTrain
    elseif mode == :last
        SaveMode_Last
    else
        throw(XfaContextException("Unknown save mode :$(mode) in DAMNIT_VARIABLES, expected :per_train or :last"))
    end
end

# A DAMNIT_VARIABLES entry: a pattern, optionally with a save mode override
struct DamnitEntry
    pattern::String
    override::Maybe{SaveMode}
end

parse_damnit_entry(pattern::AbstractString) = DamnitEntry(pattern, nothing)
parse_damnit_entry(entry::Pair{<:AbstractString, Symbol}) = DamnitEntry(entry.first, parse_save_mode(entry.second))
function parse_damnit_entry(entry)
    throw(XfaContextException("Invalid DAMNIT_VARIABLES entry $(repr(entry)), expected \"pattern\" or \"pattern\" => :mode"))
end

# The name an output is saved under in DAMNIT, which must be an identifier
damnit_name(name) = replace(name, '.' => '_')

# The names of every variable and subvariable of a context
output_names(ctx) = [collect(keys(ctx.dag)); reduce(vcat, values(ctx.subvariables); init=String[])]

# Expand DAMNIT_VARIABLES entries against the context's variables and
# subvariables into the saved outputs and their save mode overrides. Every
# pattern must match something, and a name can't get two different overrides.
function expand_saved_outputs(ctx, entries::Vector{DamnitEntry})
    outputs = output_names(ctx)
    saved = Dict{String, Maybe{SaveMode}}()
    for (; pattern, override) in entries
        re = pattern_regex(pattern)
        matches = filter(name -> occursin(re, name), outputs)
        if isempty(matches)
            throw(XfaContextException("DAMNIT_VARIABLES pattern '$(pattern)' matches no variables"))
        end

        for name in matches
            existing = get(saved, name, nothing)
            if isnothing(override)
                saved[name] = existing
            elseif isnothing(existing) || existing == override
                saved[name] = override
            else
                throw(XfaContextException("'$(name)' has conflicting save modes in DAMNIT_VARIABLES"))
            end
        end
    end

    damnit_names = Dict{String, String}()
    for name in keys(saved)
        dname = damnit_name(name)
        if haskey(damnit_names, dname)
            throw(XfaContextException("'$(name)' and '$(damnit_names[dname])' would both be saved to DAMNIT as '$(dname)'"))
        end
        damnit_names[dname] = name
    end

    saved
end

# Replace the DAMNIT_VARIABLES entries of a loaded context. Newly saved outputs
# and changed overrides take effect from the next train, which is kept as a run
# event since their data will be partial. Outputs no longer saved have their
# buffers dropped.
function set_damnit_entries!(ctx, entries::Vector{DamnitEntry})
    saved = expand_saved_outputs(ctx, entries)
    @lock ctx.run_session begin
        old = ctx.saved_outputs
        ctx.damnit_entries = entries
        ctx.saved_outputs = saved

        session = ctx.run_session[][]
        if !isnothing(session)
            filter!(kv -> haskey(saved, kv.first), session.buffers)
            changed = sort!([name for (name, override) in saved
                             if !haskey(old, name) || old[name] != override])
            if !isempty(changed)
                tid = latest_tid(session)
                message = "$(length(changed)) saved output(s) changed partway through the run"
                @warn message tid
                push!(session.events, RunEvent(tid, changed, message))
            end
        end
    end
end

# Start recording run `run` of `proposal` from train `first_tid`, called by an
# input that knows about runs. An unfinished earlier run is discarded.
function begin_run(ctx, proposal, run, first_tid)
    @lock ctx.run_session begin
        old = ctx.run_session[][]
        if !isnothing(old)
            @warn "Discarding unfinished run $(old.run) of proposal $(old.proposal)"
        end
        session = RunSession(; proposal, run, first_tid)
        ctx.run_session[][] = session
        ctx.run_ntrains[] = 0
        ctx.run_nbytes[] = 0
        Threads.atomic_xchg!(ctx.run_bounds, RunBounds(session))
    end
    @info "Recording run $(run) of proposal $(proposal) from train $(first_tid)"
end

# Stop recording the run at train `last_tid`. It's finalised once its trains
# have cleared the pipeline.
function end_run(ctx, proposal, run, last_tid)
    @lock ctx.run_session begin
        session = ctx.run_session[][]
        if isnothing(session) || (session.proposal, session.run) != (proposal, run)
            @warn "Ignoring the end of run $(run) of proposal $(proposal), it isn't being recorded"
        else
            session.last_tid = last_tid
            Threads.atomic_xchg!(ctx.run_bounds, RunBounds(session))
        end
    end
end

# Record the saved outputs among a variable's output and its subvariables, if
# its train is part of the run. Called with the session's lock held.
function record_output!(session::RunSession, saved, vd::VariableData)
    if in_run(session, vd.tid)
        if haskey(saved, vd.name)
            record_sample!(session, vd.name, saved[vd.name], vd)
        end
        for (name, out) in vd.subvariables
            if haskey(saved, name)
                record_sample!(session, name, saved[name], out)
            end
        end
    end
end

# Skipped and failed trains (no data) aren't recorded. A buffer restarts when
# the mode or the sample's shape or type changes, since per-train samples have
# to be stacked.
function record_sample!(session::RunSession, name, override, out::VariableData)
    if !isnothing(out.data)
        mode = isnothing(override) ? out.save_mode : override
        buffer = get(session.buffers, name, nothing)
        if isnothing(buffer) || buffer.mode != mode || !fits(buffer, out.data)
            if !isnothing(buffer) && buffer.mode == mode == SaveMode_PerTrain
                message = "'$(name)' changed shape or type, dropping its earlier trains"
                @warn message tid=out.tid
                push!(session.events, RunEvent(out.tid, [name], message))
            end
            buffer = SaveBuffer(mode, out.data)
            session.buffers[name] = buffer
        end
        store!(buffer, out)
    end
end

# Write the session to DAMNIT if its run has ended. Called once the pipeline
# has processed every train it will.
function finish_run!(ctx)
    session, status = @lock ctx.run_session begin
        s = ctx.run_session[][]
        if !isnothing(s) && !isnothing(s.last_tid)
            status = RunStatus(s.proposal, s.run, ctx.run_ntrains[], ctx.run_nbytes[], true)
            ctx.run_session[][] = nothing
            Threads.atomic_xchg!(ctx.run_bounds, nothing)
            s, status
        else
            nothing, nothing
        end
    end

    # Saving may have been turned off during the run
    if !isnothing(session) && ctx.damnit_enabled[] && !isempty(session.buffers)
        (; proposal, run) = session
        Threads.atomic_xchg!(ctx.saving_run, status)
        # A failed submission (e.g. damnit not installed) mustn't take down the pipeline
        try
            saved = Dict{String, Any}(damnit_name(name) => saved_value(name, buffer) for (name, buffer) in session.buffers)
            provenance = ctx.offline ? "xfa-offline" : "xfa-online"
            load_pythoncall()
            @invokelatest submit_damnit(proposal, run, saved, Dict{String, Exception}();
                                        provenance, damnit_dir=ctx.damnit_dir)
            @info "Saved run $(run) of proposal $(proposal) to DAMNIT" n=length(saved) size=Base.format_bytes(status.nbytes; binary=false)
        catch ex
            @error "Saving run $(run) of proposal $(proposal) to DAMNIT failed" exception=(ex, catch_backtrace())
        finally
            Threads.atomic_xchg!(ctx.saving_run, nothing)
        end
    end
end

# The dimensions of a sample: a DimArray's own, otherwise the output's labelled
# axes where they fit (x along a vector, y and x along an image's rows and
# columns), otherwise anonymous `dim_N` dims numbered from 0 like extra_data's.
function sample_dims(buffer::SaveBuffer)
    if !isnothing(buffer.dims)
        buffer.dims
    else
        meta = buffer.meta
        axes = if length(buffer.sample_size) == 1
            [(meta.x_axis, meta.xlabel)]
        elseif length(buffer.sample_size) == 2
            [(meta.y_axis, meta.ylabel), (meta.x_axis, meta.xlabel)]
        else
            []
        end

        ntuple(length(buffer.sample_size)) do i
            default = Symbol(:dim_, i - 1)
            if i <= length(axes) && !isnothing(axes[i][1]) && length(axes[i][1]) == buffer.sample_size[i]
                axis, label = axes[i]
                DD.Dim{isnothing(label) ? default : Symbol(label)}(collect(axis))
            else
                DD.Dim{default}()
            end
        end
    end
end

# The value saved for a buffer, sharing its memory unless its trains have to be
# sorted. Per-train samples become a
# DimArray with a trailing `trainId` dimension, named after the output unless
# the samples are named DimArrays. A SaveMode_Last sample is saved as it was,
# a plain array or value staying plain.
function saved_value(name, buffer::SaveBuffer)
    name = buffer.name isa DD.NoName ? Symbol(name) : buffer.name
    if buffer.mode == SaveMode_Last
        if !buffer.is_array
            only(buffer.data)
        elseif isnothing(buffer.dims)
            reshape(buffer.data, buffer.sample_size)
        else
            DD.DimArray(reshape(buffer.data, buffer.sample_size), buffer.dims;
                        refdims=buffer.refdims, name, metadata=buffer.metadata)
        end
    else
        data = reshape(buffer.data, buffer.sample_size..., length(buffer.tids))
        tids = buffer.tids
        # Trains can be fed out of order, sort them (copying) if they were
        if !issorted(tids)
            perm = sortperm(tids)
            data = data[ntuple(Returns(:), length(buffer.sample_size))..., perm]
            tids = tids[perm]
        end
        dims = (sample_dims(buffer)..., DD.Dim{:trainId}(tids))
        DD.DimArray(data, dims; refdims=buffer.refdims, name, metadata=buffer.metadata)
    end
end

# Write a run's saved values to DAMNIT, implemented by the PythonCall extension
function submit_damnit end
