using CurveFit: NonlinearCurveFitProblem, CurveFitProblem, LinearCurveFitAlgorithm,
    ScalarModel, solve
using CurveFit.SciMLBase: successful_retcode
using SpecialFunctions: erf as base_erf
using NaNStatistics: nanmean
using FFTW: rfft, rfftfreq
using Statistics: median

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

# Estimate the dominant period of (x, y) via FFT of the mean-subtracted signal
# with quadratic interpolation around the peak bin for sub-bin resolution.
# Handles non-uniform sampling by linearly interpolating onto a uniform grid
# with spacing `median(diff(x))` before the FFT. Returns period in units of x;
# falls back to the full span when the data is too short for a meaningful FFT.
function estimate_period(x::AbstractVector, y::AbstractVector, y0::Real)
    span = abs(maximum(x) - minimum(x))
    fallback = span > 0 ? span : 1.0
    n = length(y)
    if n < 4 || span == 0
        return fallback
    end

    # Sort by x (samples may arrive out of order) and resample onto a uniform
    # grid at the median spacing so the FFT sees evenly-spaced data.
    perm = sortperm(x)
    xs = x[perm]
    ys = y[perm]
    dx = median(diff(xs))
    if !(dx > 0)
        return fallback
    end
    n = max(4, round(Int, span / dx) + 1)
    xu = range(xs[1], xs[end]; length=n)
    yu = similar(ys, n)
    j = 1
    for (i, xi) in enumerate(xu)
        while j < length(xs) - 1 && xs[j + 1] < xi
            j += 1
        end
        t = (xi - xs[j]) / (xs[j + 1] - xs[j])
        yu[i] = ys[j] + t * (ys[j + 1] - ys[j])
    end

    spectrum = abs.(rfft(yu .- y0))
    # Bin 0 is DC (already removed, but skip to be safe).
    idx = argmax(@view spectrum[2:end]) + 1
    freqs = rfftfreq(n, (n - 1) / span)
    # Quadratic interpolation around the peak for sub-bin frequency resolution.
    δ = 0.0
    if 1 < idx < length(spectrum)
        a, b, c = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
        denom = a - 2b + c
        if denom != 0
            δ = clamp(0.5 * (a - c) / denom, -0.5, 0.5)
        end
    end
    df = length(freqs) > 1 ? freqs[2] - freqs[1] : 0.0
    f = freqs[idx] + δ * df
    return f > 0 ? 1 / f : fallback
end

# Sinusoid: y0 offset, amplitude A, period (in units of x), phase φ (in units
# of x — a horizontal shift, not an angle).
function sinusoid(x, y0, A, period, φ)
    return y0 + A * sin(2π * (x - φ) / period)
end

# Fit a sinusoid to (xdata, ydata). Returns `(popt, retcode)` where popt is
# [y0, A, period, φ] (or `nothing` on failure).
function fit_sin(ydata::AbstractVector, xdata::Maybe{AbstractVector}=nothing;
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

    # Center x around 0 for the fit: period and φ are otherwise strongly
    # correlated (a tiny period change → large phase shift at large x), which
    # makes the nonlinear solver drift along that valley. Undo the shift on
    # popt afterwards.
    x_center = (maximum(x) + minimum(x)) / 2
    x_fit = x .- x_center

    if !isnothing(p0)
        # Convert caller-supplied p0 (original x frame) to the shifted frame.
        p0 = copy(p0)
        p0[4] -= x_center
    end

    if isnothing(p0)
        y0_guess = nanmean(y)
        T = estimate_period(x_fit, y, y0_guess)
        # With period fixed, the model is linear in (y0, a, b) where
        # A·sin(2π(x-φ)/T) = a·sin(2πx/T) + b·cos(2πx/T). Solve that LS to
        # seed the nonlinear fit near the optimum.
        s = sin.(2π .* x_fit ./ T)
        c = cos.(2π .* x_fit ./ T)
        M = hcat(ones(length(x_fit)), s, c)
        coeffs = M \ y
        y0, a, b = coeffs
        A = hypot(a, b)
        # a = A·cos(2πφ/T), b = -A·sin(2πφ/T) ⇒ φ = -T/(2π)·atan(b, a)
        φ = -T / (2π) * atan(b, a)
        p0 = [y0, A, T, φ]
    end

    model = ScalarModel((p, xi) -> sinusoid(xi, p[1], p[2], p[3], p[4]))
    popt, retcode = run_fit(model, p0, x_fit, y; sigma=σ)
    if isnothing(popt)
        return nothing, retcode
    end

    # Undo the x shift: φ in the original frame is offset by x_center.
    popt[4] += x_center

    # Canonicalize: A ≥ 0, period > 0, φ ∈ [0, period).
    if popt[3] < 0
        popt[3] = -popt[3]
        popt[2] = -popt[2]
    end
    if popt[2] < 0
        popt[2] = -popt[2]
        popt[4] += popt[3] / 2
    end
    popt[4] = mod(popt[4], popt[3])
    return popt, retcode
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
