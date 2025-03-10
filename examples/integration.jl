using TaylorDiff: TaylorDiff, TaylorScalar, make_seed, flatten, get_coefficient,
                  set_coefficient, append_coefficient
using TaylorSeries, TaylorIntegration
using ODEProblemLibrary, OrdinaryDiffEq, BenchmarkTools, Symbolics
SymbolicUtils.ENABLE_HASHCONSING[] = true

# There are two ways to compute the Taylor coefficients of a ODE solution
# 1. Using naive repeated differentiation
# 2. First simplify the expansion using Symbolics and then evaluate the expression

"""
# The first method

For ODE u' = f(u, p, t) and initial condition (u0, t0), computes Taylor expansion of the solution `u` up to order `P` using repeated differentiation.
"""
function jetcoeffs(f::ODEFunction{iip}, u0, p, t0, ::Val{P}) where {P, iip}
    t = TaylorScalar{P}(t0, one(t0))
    u = make_seed(u0, zero(u0), Val(P))
    fu = copy(u)
    for index in 1:P
        if iip
            f(fu, u, p, t)
        else
            fu = f(u, p, t)
        end
        d = get_coefficient(fu, index - 1) / index
        u = set_coefficient(u, index, d)
    end
    u
end

function scalar_test()
    P = 6
    prob = prob_ode_linear
    t0 = prob.tspan[1]
    # TaylorIntegration test
    ts = t0 + Taylor1(typeof(t0), P)
    u_ts = Taylor1(prob.u0, P)
    @btime TaylorIntegration.jetcoeffs!($prob.f, $ts, $u_ts, $prob.p)

    # TaylorDiff test
    @btime jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
    u_td = jetcoeffs(prob.f, prob.u0, prob.p, t0, Val(P))
    @assert u_ts.coeffs ≈ collect(flatten(u_td))
end

function array_test()
    P = 6
    prob = prob_ode_lotkavolterra
    t0 = prob.tspan[1]
    # TaylorIntegration test
    ts = t0 + Taylor1(typeof(t0), P)
    u_ts = [Taylor1(x, P) for x in prob.u0]
    du = similar(u_ts)
    uaux = similar(u_ts)
    @btime TaylorIntegration.jetcoeffs!($prob.f, $ts, $u_ts, $du, $uaux, $prob.p)

    # TaylorDiff test
    @btime jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
    u_td = jetcoeffs(prob.f, prob.u0, prob.p, t0, Val(P))
    for i in eachindex(u_ts)
        @assert u_ts[i].coeffs ≈ collect(flatten(u_td[i]))
    end
end

"""
# The second method

For ODE u' = f(u, p, t) and initial condition (u0, t0), symbolically computes Taylor expansion of the solution `u` up to order `P`, and then builds a function to evaluate the expression.
"""
function build_jetcoeffs(f::ODEFunction{iip}, p, ::Val{P}, length = nothing) where {P, iip}
    @variables t0::Real
    u0 = isnothing(length) ? Symbolics.variable(:u0) : Symbolics.variables(:u0, 1:length)
    if iip
        @assert length isa Integer
        f0 = similar(u0)
        f(f0, u0, p, t0)
    else
        f0 = f(u0, p, t0)
    end
    u = TaylorDiff.make_seed(u0, f0, Val(1))
    for index in 2:P
        t = TaylorScalar{index - 1}(t0, one(t0))
        if iip
            fu = similar(u)
            f(fu, u, p, t)
        else
            fu = f(u, p, t)
        end
        d = get_coefficient(fu, index - 1) / index
        u = append_coefficient(u, d)
    end
    build_function(u, u0, t0; expression = Val(false), cse = true)
end

function simplify_scalar_test()
    P = 6
    prob = prob_ode_linear
    t0 = prob.tspan[1]
    @btime jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
    fast_jetcoeffs = build_jetcoeffs(prob.f, prob.p, Val(P))
    @btime $fast_jetcoeffs($prob.u0, $t0)
end

function simplify_array_test()
    P = 6
    prob = prob_ode_lotkavolterra
    t0 = prob.tspan[1]
    @btime jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
    fast_oop, fast_iip = build_jetcoeffs(prob.f, prob.p, Val(P), length(prob.u0))
    @btime $fast_oop($prob.u0, $t0)
end

@generated function evaluate_polynomial(t::TaylorScalar{T, P}, z) where {T, P}
    ex = :(v[$(P + 1)])
    for i in P:-1:1
        ex = :(v[$i] + z * $ex)
    end
    return :($(Expr(:meta, :inline)); v = flatten(t); $ex)
end
