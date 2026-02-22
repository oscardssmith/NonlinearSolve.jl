"""
    ModAB()

ModAB (Modified Anderson-Bjork)

Use the [ModAB method](https://iopscience.iop.org/article/10.1088/1757-899X/1276/1/012010/) to find a root of a bracketed
function, with a convergence rate between 1 and 1.62.

This method was introduced in the paper "Modified Anderson-Bjork’s method for solving non-linear equations 
in structural mechanics" (https://doi.org/10.1088/1757-899X/1276/1/012010) by
N Ganchovski and A Traykov.
"""
struct ModAB <: AbstractBracketingAlgorithm
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::ModAB, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`ModAB` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    x1, x2 = minmax(promote(prob.tspan...)...)
    y1, y2 = f(x1), f(x2)

    abstol = NonlinearSolveBase.get_tolerance(
        x1, abstol, promote_type(eltype(x1), eltype(x2))
    )

    if iszero(y1)
        return build_exact_solution(prob, alg, x1, y1, ReturnCode.ExactSolutionLeft)
    end

    if iszero(y2)
        return build_exact_solution(prob, alg, x2, y2, ReturnCode.ExactSolutionRight)
    end

    if sign(y1) == sign(y2)
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, x1, y1, x1, x2, ReturnCode.InitialFailure)
    end

    bisecting = true
    threshold = 0.0
    C = 16 # safety factor of 4 iteration behind the threshold before reset to bisection
    side = 0 # tracks the side that has moved at the previous iteration
    ϵ = abstol
    x0 = x1
    i = 1
    while i < maxiters
        local x3, y3
        if bisecting # Bisection
            x3 = (x1 + x2) / 2
            y3 = f(x3)
            # Ordinate of chord at midpoint
            ym = (y1 + y2) / 2
            y1a = abs(y1)
            y2a = abs(y2)
            r = y1a > 0 && y2a > 0 ? min(y1a, y2a) / max(y1a, y2a) : 1
            k = r^0.25 # Factor for limiting deviation from straight line
            if abs(ym - y3) < k * (abs(ym) + abs(y3))
                bisecting = false
                threshold = (x2 - x1) * C
            end
        else # Falsi
            x3 = (x1 * y2 - y1 * x2) / (y2 - y1)
            y3 = f(x3)
            threshold /= 2
        end
        if iszero(y3)
            return build_exact_solution(prob, alg, x3, y3, ReturnCode.Success)
        elseif abs(x3 - x0) <= ϵ
            return build_bracketing_solution(prob, alg, x3, y3, x1, x2, ReturnCode.Success)
        end
        x0 = x3
        if sign(y1) == sign(y3)
            if side == 1
                m = 1 - y3 / y1
                if m <= 0
                    y2 /= 2
                else
                    y2 *= m
                end
            elseif !bisecting
                side = 1
            end
            x1, y1 = x3, y3
        else
            if side == -1
                m = 1 - y3 / y2
                if m <= 0
                    y1 /= 2
                else
                    y1 *= m
                end
            elseif !bisecting
                side = -1
            end
            x2, y2 = x3, y3
        end
        if nextfloat(x1) == x2
            return build_bracketing_solution(prob, alg, x2, f(x2), x1, x2, ReturnCode.FloatingPointLimit)
        end
        i += 1
        if x2 - x1 > threshold # if AB fails to shrink the interval enough
            bisecting = true
            side = 0
        end
    end
    return build_bracketing_solution(prob, alg, x1, y1, x1, x2, ReturnCode.MaxIters)
end