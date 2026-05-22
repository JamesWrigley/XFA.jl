using CurveFit: NonlinearCurveFitProblem, CurveFitProblem, LinearCurveFitAlgorithm,
    ScalarModel, solve
using CurveFit.SciMLBase: successful_retcode
using SpecialFunctions: erf as base_erf
using NaNStatistics: nanmean

# Unnormalized Gaussian profile.
function gaussian(x, y0, A, μ, σ)
    return y0 + A * exp(-(x - μ)^2 / (2 * σ^2))
end

# Parameterized error function. `fwhm` is the Gaussian FWHM; `A` is the
# total step height between the two asymptotes.
function erf(x, y0, A, center, fwhm)
    return (A / 2) * base_erf(2√log(2) / fwhm * (x - center)) + y0
end

# Default xdata, mask non-finite ydata samples (and non-positive sigma entries
# when sigma is provided), return paired (x, y, sigma_or_nothing) as Float64
# vectors. Returns `nothing` if no finite samples remain.
function mask_finite(ydata::AbstractVector, xdata::Maybe{AbstractVector},
                     sigma::Maybe{AbstractVector}=nothing)
    if isnothing(xdata)
        xdata = eachindex(ydata)
    end
    mask = isfinite.(ydata)
    if !isnothing(sigma)
        mask .&= isfinite.(sigma) .& (sigma .> 0)
    end
    x = float.(xdata[mask])
    y = float.(ydata[mask])
    if isempty(y)
        return nothing
    end
    σ = isnothing(sigma) ? nothing : float.(sigma[mask])
    return x, y, σ
end

# Solve a nonlinear least-squares fit, returning `(popt_or_nothing, retcode)`
# where retcode is the SciML return code as a Symbol.
function run_fit(model, p0, x, y; sigma=nothing, lb=nothing, ub=nothing)
    prob = NonlinearCurveFitProblem(model, collect(p0), x, y, sigma; lb, ub)
    sol = solve(prob)

    u = successful_retcode(sol.retcode) ? sol.u : nothing
    return u, Symbol(sol.retcode)
end

# Fit a Gaussian to (xdata, ydata). Returns `(popt, retcode)` where popt is the
# fitted [y0, A, μ, σ] vector (or `nothing` on failure) and retcode is a Symbol
# tagged with the solver outcome — `:NoFiniteSamples` if there were no usable
# samples, otherwise the SciML return code.
#
# `A_sign`: 0 (default) allows either peak direction, 1 forces upward, -1 forces downward.
function fit_gaussian(ydata::AbstractVector, xdata::Maybe{AbstractVector}=nothing;
                      sigma::Maybe{AbstractVector}=nothing,
                      p0::Maybe{AbstractVector}=nothing, A_sign::Int=0)
    if !isnothing(p0) && length(p0) != 4
        throw(ArgumentError("p0 must have length 4, got length $(length(p0))"))
    end

    masked = mask_finite(ydata, xdata, sigma)
    if isnothing(masked)
        return nothing, :NoFiniteSamples
    end
    x, y, σ = masked

    if isnothing(p0)
        if A_sign >= 0
            μ_idx = argmax(y)
            A = max(1.0, y[μ_idx])
            y0 = minimum(y)
        else
            μ_idx = argmin(y)
            A = min(-1.0, y[μ_idx])
            y0 = maximum(y)
        end
        μ = x[μ_idx]
        width = abs(maximum(x) - minimum(x)) / 4
        p0 = [y0, A, μ, width]
    end

    lb, ub = nothing, nothing
    if A_sign != 0
        Amin, Amax = A_sign > 0 ? (0.0, Inf) : (-Inf, 0.0)
        lb = [-Inf, Amin, -Inf, 0.0]
        ub = [Inf, Amax, Inf, Inf]
    end

    model = ScalarModel((p, xi) -> gaussian(xi, p[1], p[2], p[3], p[4]))
    popt, retcode = run_fit(model, p0, x, y; sigma=σ, lb, ub)
    if isnothing(popt)
        return nothing, retcode
    end

    # σ can converge to a negative value (only σ² appears in the model).
    popt[4] = abs(popt[4])
    return popt, retcode
end

# Fit a parameterized error function to (xdata, ydata). Returns `(popt,
# retcode)` — popt is the fitted [y0, A, center, width] vector (or `nothing` on
# failure) and retcode is a Symbol tagged with the solver outcome.
function fit_erf(ydata::AbstractVector, xdata::Maybe{AbstractVector}=nothing;
                 sigma::Maybe{AbstractVector}=nothing,
                 p0::Maybe{AbstractVector}=nothing)
    if !isnothing(p0) && length(p0) != 4
        throw(ArgumentError("p0 must have length 4, got length $(length(p0))"))
    end

    masked = mask_finite(ydata, xdata, sigma)
    if isnothing(masked)
        return nothing, :NoFiniteSamples
    end
    x, y, σ = masked

    if isnothing(p0)
        # Edges estimated from the outer 10% of the data on each side.
        ten_pct = max(1, length(y) ÷ 10)
        left_edge = nanmean(@view y[1:ten_pct])
        right_edge = nanmean(@view y[end - ten_pct + 1:end])

        A = right_edge - left_edge
        center = nanmean(x)
        width = abs(maximum(x) - minimum(x)) / 4
        y0 = nanmean(y)
        p0 = [y0, A, center, width]
    end

    model = ScalarModel((p, xi) -> erf(xi, p[1], p[2], p[3], p[4]))
    return run_fit(model, p0, x, y; sigma=σ)
end

# Fit a line to (xdata, ydata). Returns `(popt, retcode)` — popt is
# [slope, intercept] (or `nothing` on failure / fewer than two finite samples)
# and retcode is a Symbol tagged with the solver outcome.
function fit_line(ydata::AbstractVector, xdata::Maybe{AbstractVector}=nothing;
                  sigma::Maybe{AbstractVector}=nothing)
    masked = mask_finite(ydata, xdata, sigma)
    if isnothing(masked)
        return nothing, :NoFiniteSamples
    end
    x, y, σ = masked

    if length(y) < 2
        return nothing, :InsufficientData
    end

    sol = solve(CurveFitProblem(x, y; sigma=σ), LinearCurveFitAlgorithm())
    retcode = Symbol(sol.retcode)
    u = successful_retcode(sol.retcode) ? collect(sol.u) : nothing
    return u, retcode
end
