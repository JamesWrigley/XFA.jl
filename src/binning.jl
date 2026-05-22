# Accumulating pair sequence for binning scan data. Bins are fixed regions of
# width `resolution` keyed by floor(x / resolution), so sample-to-bin assignment
# is independent of arrival order.
Base.@kwdef mutable struct AccuPairSequence
    resolution::Float64

    # Parallel arrays sorted by bin_keys ascending. bin_keys is the fixed
    # integer bin index; x_values/y_values hold each bin's running x and y
    # mean; y_lower/y_upper hold y_mean ± half-std (for plotting).
    bin_keys::Vector{Int} = Int[]
    x_values::Vector{Float64} = Float64[]
    y_values::Vector{Float64} = Float64[]
    y_lower::Vector{Float64} = Float64[]
    y_upper::Vector{Float64} = Float64[]
    sigma::Vector{Float64} = Float64[]

    # Per-bin Welford state (counts and sum of squared y-deviations).
    counts::Vector{Int} = Int[]
    y_m2::Vector{Float64} = Float64[]
end

function AccuPairSequence(resolution::Real)
    if resolution <= 0
        throw(ArgumentError("resolution must be positive"))
    end
    AccuPairSequence(; resolution=Float64(resolution))
end

function AccuPairSequence(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real},
                          resolution::Real)
    if length(xs) != length(ys)
        throw(ArgumentError("xs and ys must have the same length " *
                            "(got $(length(xs)), $(length(ys)))"))
    end
    seq = AccuPairSequence(resolution)
    for i in eachindex(xs, ys)
        append!(seq, xs[i], ys[i])
    end
    return seq
end

bin_key(seq::AccuPairSequence, x::Real) = floor(Int, x / seq.resolution)

function Base.append!(seq::AccuPairSequence, x::Real, y::Real)
    key = bin_key(seq, x)
    i = searchsortedfirst(seq.bin_keys, key)
    if i <= length(seq.bin_keys) && seq.bin_keys[i] == key
        # Welford merge into existing bin.
        seq.counts[i] += 1
        n = seq.counts[i]
        seq.x_values[i] += (x - seq.x_values[i]) / n
        prev_y_mean = seq.y_values[i]
        seq.y_values[i] += (y - prev_y_mean) / n
        seq.y_m2[i] += (y - prev_y_mean) * (y - seq.y_values[i])
        half_std = 0.5 * sqrt(seq.y_m2[i] / n)
        seq.y_lower[i] = seq.y_values[i] - half_std
        seq.y_upper[i] = seq.y_values[i] + half_std
        seq.sigma[i] = 1 / sqrt(n)
    else
        # New bin at the sorted insertion point; single-sample band collapses
        # to the line (half_std == 0) until a second sample arrives.
        insert!(seq.bin_keys, i, key)
        insert!(seq.x_values, i, x)
        insert!(seq.y_values, i, y)
        insert!(seq.y_lower, i, y)
        insert!(seq.y_upper, i, y)
        insert!(seq.sigma, i, 1.0)
        insert!(seq.counts, i, 1)
        insert!(seq.y_m2, i, 0.0)
    end
    return seq
end

function reset!(seq::AccuPairSequence)
    empty!(seq.bin_keys)
    empty!(seq.x_values)
    empty!(seq.y_values)
    empty!(seq.y_lower)
    empty!(seq.y_upper)
    empty!(seq.sigma)
    empty!(seq.counts)
    empty!(seq.y_m2)
    return seq
end

Base.length(seq::AccuPairSequence) = length(seq.x_values)
