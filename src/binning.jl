# Accumulating pair sequence for binning scan data.
mutable struct AccuPairSequence
    resolution::Float64

    # Arrays sorted by x_values ascending. x_values/y_values hold each bin's
    # running x and y mean; y_lower/y_upper hold y_mean ± half-std.
    x_values::Vector{Float64}
    y_values::Vector{Float64}
    y_lower::Vector{Float64}
    y_upper::Vector{Float64}

    # Per-bin Welford state (counts and sum of squared y-deviations).
    counts::Vector{Int}
    y_m2::Vector{Float64}
end

function AccuPairSequence(resolution::Real)
    if resolution <= 0
        throw(ArgumentError("resolution must be positive"))
    end

    AccuPairSequence(Float64(resolution),
                     Float64[], Float64[], Float64[], Float64[],
                     Int[], Float64[])
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

# Return the index of the bin whose x_value is within resolution of `x`, or
# nothing if none. Bins are pairwise > resolution apart, so at most one
# candidate qualifies — check the two nearest via binary search.
function find_matching_bin(seq::AccuPairSequence, x::Real)
    n = length(seq.x_values)
    if n == 0
        return nothing
    end
    i = searchsortedfirst(seq.x_values, x)
    best_idx = nothing
    best_dist = seq.resolution
    for j in (i - 1, i)
        if 1 <= j <= n
            d = abs(x - seq.x_values[j])
            if d <= best_dist
                best_dist = d
                best_idx = j
            end
        end
    end
    return best_idx
end

function Base.append!(seq::AccuPairSequence, x::Real, y::Real)
    j = find_matching_bin(seq, x)
    if isnothing(j)
        # New bin at the sorted insertion point; single-sample band collapses
        # to the line (half_std == 0).
        j = searchsortedfirst(seq.x_values, x)
        insert!(seq.x_values, j, x)
        insert!(seq.y_values, j, y)
        insert!(seq.y_lower, j, y)
        insert!(seq.y_upper, j, y)
        insert!(seq.counts, j, 1)
        insert!(seq.y_m2, j, 0.0)
    else
        # Welford merge. x_mean drift is bounded by resolution and neighbours
        # are > resolution away, so the bin keeps its sorted position.
        seq.counts[j] += 1
        n = seq.counts[j]
        seq.x_values[j] += (x - seq.x_values[j]) / n
        prev_y_mean = seq.y_values[j]
        seq.y_values[j] += (y - prev_y_mean) / n
        seq.y_m2[j] += (y - prev_y_mean) * (y - seq.y_values[j])
        half_std = 0.5 * sqrt(seq.y_m2[j] / n)
        seq.y_lower[j] = seq.y_values[j] - half_std
        seq.y_upper[j] = seq.y_values[j] + half_std
    end
    return seq
end

function reset!(seq::AccuPairSequence)
    empty!(seq.x_values)
    empty!(seq.y_values)
    empty!(seq.y_lower)
    empty!(seq.y_upper)
    empty!(seq.counts)
    empty!(seq.y_m2)
    return seq
end

Base.length(seq::AccuPairSequence) = length(seq.x_values)
