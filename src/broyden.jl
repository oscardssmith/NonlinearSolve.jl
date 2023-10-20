# Sadly `Broyden` is taken up by SimpleNonlinearSolve.jl
"""
    GeneralBroyden(max_resets, linesearch)
    GeneralBroyden(; max_resets = 3, linesearch = LineSearch())

An implementation of `Broyden` with reseting and line search.

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`. It is
    recommended to use [LiFukushimaLineSearchCache](@ref) -- a derivative free linesearch
    specifically designed for Broyden's method.
"""
@concrete struct GeneralBroyden <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
    linesearch
end

function GeneralBroyden(; max_resets = 3, linesearch = LineSearch())
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return GeneralBroyden(max_resets, linesearch)
end

@concrete mutable struct GeneralBroydenCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    du
    fu
    fu2
    dfu
    p
    J⁻¹
    J⁻¹₂
    J⁻¹df
    force_stop::Bool
    resets::Int
    max_resets::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    prob
    stats::NLStats
    lscache
end

get_fu(cache::GeneralBroydenCache) = cache.fu

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::GeneralBroyden, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu = evaluate_f(prob, u)
    J⁻¹ = __init_identity_jacobian(u, fu)
    return GeneralBroydenCache{iip}(f, alg, u, _mutable_zero(u), fu, zero(fu),
        zero(fu), p, J⁻¹, zero(fu'), _mutable_zero(u), false, 0, alg.max_resets,
        maxiters, internalnorm, ReturnCode.Default, abstol, prob, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)))
end

function perform_step!(cache::GeneralBroydenCache{true})
    @unpack f, p, du, fu, fu2, dfu, u, J⁻¹, J⁻¹df, J⁻¹₂ = cache
    T = eltype(u)

    mul!(du, J⁻¹, -fu)
    α = perform_linesearch!(cache.lscache, u, du)
    axpy!(α, du, u)
    f(fu2, u, p)

    cache.internalnorm(fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the inverse jacobian
    dfu .= fu2 .- fu
    if cache.resets < cache.max_resets &&
       (all(x -> abs(x) ≤ 1e-12, du) || all(x -> abs(x) ≤ 1e-12, dfu))
        fill!(J⁻¹, 0)
        J⁻¹[diagind(J⁻¹)] .= T(1)
        cache.resets += 1
    else
        mul!(J⁻¹df, J⁻¹, dfu)
        mul!(J⁻¹₂, du', J⁻¹)
        du .= (du .- J⁻¹df) ./ (dot(du, J⁻¹df) .+ T(1e-5))
        mul!(J⁻¹, reshape(du, :, 1), J⁻¹₂, 1, 1)
    end
    fu .= fu2

    return nothing
end

function perform_step!(cache::GeneralBroydenCache{false})
    @unpack f, p = cache
    T = eltype(cache.u)

    cache.du = cache.J⁻¹ * -cache.fu
    α = perform_linesearch!(cache.lscache, cache.u, cache.du)
    cache.u = cache.u .+ α * cache.du
    cache.fu2 = f(cache.u, p)

    cache.internalnorm(cache.fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the inverse jacobian
    cache.dfu = cache.fu2 .- cache.fu
    if cache.resets < cache.max_resets &&
       (all(x -> abs(x) ≤ 1e-12, cache.du) || all(x -> abs(x) ≤ 1e-12, cache.dfu))
        J⁻¹ = similar(cache.J⁻¹)
        fill!(J⁻¹, 0)
        J⁻¹[diagind(J⁻¹)] .= T(1)
        cache.J⁻¹ = J⁻¹
        cache.resets += 1
    else
        cache.J⁻¹df = cache.J⁻¹ * cache.dfu
        cache.J⁻¹₂ = cache.du' * cache.J⁻¹
        cache.du = (cache.du .- cache.J⁻¹df) ./ (dot(cache.du, cache.J⁻¹df) .+ T(1e-5))
        cache.J⁻¹ = cache.J⁻¹ .+ cache.du * cache.J⁻¹₂
    end
    cache.fu = cache.fu2

    return nothing
end

function SciMLBase.reinit!(cache::GeneralBroydenCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
