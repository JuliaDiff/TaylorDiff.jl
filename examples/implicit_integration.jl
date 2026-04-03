include("integration.jl")

using NonlinearSolve, ODEProblemLibrary, LinearAlgebra

# a simple fixed time-step driver for rapid algorithm prototyping
function integrate(prob, solver; h)
    cache = prepare(prob, solver)
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        update!(u, prob, solver, cache, t, h)
        t += h
    end
    return u
end

# the μ-Taylor solver in NASA paper
struct MuTaylor{P, T}
    order::Val{P}
    μ::T
end

function prepare(prob, solver::MuTaylor{P, T}) where {T, P}
    propagator_all, _ = build_propagator(prob.f, prob.p, ntuple(_ -> 1.0, solver.order), length(prob.u0))
    propagator, _ = propagator_all
    return (; propagator)
end

function update!(u, prob, solver::MuTaylor{P, T}, cache, t, h) where {T, P}
    (; μ) = solver
    (; propagator) = cache
    tm = t + μ * h
    nlf = (u_next, u_curr) -> propagator(u_next, tm, -μ * h) .- u_curr
    nlprob = NonlinearProblem(nlf, copy(u), u)
    sol = solve(nlprob)
    return u .= real(propagator(sol.u, tm, (1 - μ) * h))
end

struct MuTaylorExtrapolation{P, T1, T2, T3, T4}
    order::Val{P}
    μ1::T1
    c1::T2
    μ2::T3
    c2::T4
end

function prepare(prob, solver::MuTaylorExtrapolation{P, T1, T2, T3, T4}) where {P, T1, T2, T3, T4}
    propagator_all, _ = build_propagator(prob.f, prob.p, ntuple(_ -> 1.0, solver.order), length(prob.u0))
    propagator, _ = propagator_all
    return (; propagator)
end

function update!(u, prob, solver::MuTaylorExtrapolation{P, T1, T2, T3, T4}, cache, t, h) where {T1, T2, T3, T4, P}
    (; μ1, c1, μ2, c2) = solver
    (; propagator) = cache
    tm1 = t + μ1 * h
    tm2 = t + μ2 * h
    nlf = (u_next, u_curr) -> propagator(u_next, tm1, -μ1 * h) - u_curr
    nlprob1 = NonlinearProblem(nlf, copy(u), u)
    sol1 = solve(nlprob1)
    u_trial1 = real(propagator(sol1.u, tm1, (1 - μ1) * h))
    nlf2 = (u_next, u_curr) -> propagator(u_next, tm2, -μ2 * h) - u_curr
    nlprob2 = NonlinearProblem(nlf2, copy(u), u)
    sol2 = solve(nlprob2)
    u_trial2 = real(propagator(sol2.u, tm2, (1 - μ2) * h))
    return u .= (c1 * u_trial1 + c2 * u_trial2)
end

function estimate_order(solver, prob; dt1 = 1/12, dt2 = 1/16)
    ref = prob.f.analytic(prob.u0, prob.p, prob.tspan[2])
    println("Reference: ", ref)
    u1 = integrate(prob, solver; h = dt1)
    println("u1: ", u1)
    u2 = integrate(prob, solver; h = dt2)
    println("u2: ", u2)
    err1 = norm(u1 - ref)
    err2 = norm(u2 - ref)
    order = log(err1 / err2) / log(dt1 / dt2)
    return order
end

u0 = rand(2)
linear_f = ODEFunction(
    ODEProblemLibrary.f_2dlinear, analytic = ODEProblemLibrary.f_2dlinear_analytic
)
prob = ODEProblem(linear_f, u0, (0.0, 1.0), 1.01)
prob_complex = ODEProblem(
    linear_f, complex(u0), (0.0, 1.0), 1.01
)

ImplicitTaylor = MuTaylor(Val(1), 1.0)
ImplicitTaylorMidpoint = MuTaylor(Val(1), 0.5)
estimate_order(ImplicitTaylor, prob)
estimate_order(ImplicitTaylorMidpoint, prob)

ImplicitTaylor2 = MuTaylor(Val(2), 1.0)
ImplicitTaylor2Midpoint = MuTaylor(Val(2), 0.5)
ImplicitTaylor2Complex = MuTaylor(Val(2), 0.5 + sqrt(3) * im / 6)
estimate_order(ImplicitTaylor2, prob)
estimate_order(ImplicitTaylor2Midpoint, prob)
estimate_order(ImplicitTaylor2Complex, prob_complex)

ImplicitTaylor3Midpoint = MuTaylor(Val(3), 0.5)
ImplicitTaylor3Extrapolated = MuTaylorExtrapolation(Val(3), 0.5, 0.8, 0.5 + 0.5im, 0.2)
estimate_order(ImplicitTaylor3Midpoint, prob)
estimate_order(ImplicitTaylor3Extrapolated, prob_complex; dt1 = 1 / 12, dt2 = 1 / 16)

ImplicitTaylor4Midpoint = MuTaylor(Val(4), 0.5)
ImplicitTaylor4Extrapolated = MuTaylorExtrapolation(
    Val(4),
    0.5 + 0.5im * tan(π / 10), tan(π / 10) * sec(π / 10)^5,
    0.5 + 0.5im * tan(3π / 10), tan(3π / 10) * sec(3π / 10)^5
)
estimate_order(ImplicitTaylor4Midpoint, prob)
estimate_order(ImplicitTaylor4Extrapolated, prob_complex; dt1 = 1 / 12, dt2 = 1 / 16)

function normalized_pade(p::Int, q::Int)
    RI = Rational{BigInt}
    N = p + q
    c = Vector{RI}(undef, N + 1)
    c[1] = big(1) // big(1)
    for k in 1:N
        c[k + 1] = c[k] // big(k)
    end
    if q == 0
        dres = RI[]
    else
        A = zeros(RI, q, q)
        rhs = zeros(RI, q)
        for i in 1:q
            k = p + i                     # 对应 z^k 阶 (0-indexed)
            rhs[i] = -c[k + 1]             # -c_k
            for j in 1:q
                idx = k - j               # 0-indexed Taylor index
                A[i, j] = idx >= 0 ? c[idx + 1] : big(0) // big(1)
            end
        end
        dres = A \ rhs                       # Rational 精确求解
    end
    n = zeros(RI, p + 1)
    for k in 0:p
        n[k + 1] = c[k + 1]
        for j in 1:min(k, q)
            n[k + 1] += dres[j] * c[k - j + 1]
        end
    end
    d = vcat(big(1) // big(1), dres)
    normalized_n = [x * factorial(k - 1) for (k, x) in enumerate(n)]
    normalized_d = [x * factorial(k - 1) for (k, x) in enumerate(d)]
    return normalized_n, normalized_d
end

# the μ-Taylor solver in NASA paper
struct TaylorPade{P, Q}
    n_order::Val{P}
    d_order::Val{Q}
end

function prepare(prob, solver::TaylorPade{P, Q}) where {P, Q}
    polynomial_p, polynomial_q = normalized_pade(P, Q)
    tuple_p = Base.tail(tuple(map(Float64, polynomial_p)...))
    tuple_q = Base.tail(tuple(map(Float64, polynomial_q)...))
    println("Pade coefficients (numerator): ", tuple_p)
    println("Pade coefficients (denominator): ", tuple_q)
    propagator_p_all, _ = build_propagator(prob.f, prob.p, tuple_p, length(prob.u0))
    propagator_q_all, _ = build_propagator(prob.f, prob.p, tuple_q, length(prob.u0))
    propagator_p, _ = propagator_p_all
    propagator_q, _ = propagator_q_all
    return (; propagator_p, propagator_q)
end

function update!(u, prob, solver::TaylorPade{P, Q}, cache, t, h) where {P, Q}
    (; propagator_p, propagator_q) = cache
    rhs = propagator_p(u, t, h)
    nlf = (u_next, rhs) -> propagator_q(u_next, t + h, h) .- rhs
    nlprob = NonlinearProblem(nlf, copy(u), rhs)
    sol = solve(nlprob)
    return u .= sol.u
end

# diagonal: Taylor-Gauss solvers
TaylorPade1v1 = TaylorPade(Val(1), Val(1))
TaylorPade2v2 = TaylorPade(Val(2), Val(2))
TaylorPade3v3 = TaylorPade(Val(3), Val(3))
estimate_order(TaylorPade1v1, prob)
estimate_order(TaylorPade2v2, prob)
estimate_order(TaylorPade3v3, prob)

# sub-diagonal: Taylor-Radau solvers
TaylorPade0v1 = TaylorPade(Val(0), Val(1))
TaylorPade1v2 = TaylorPade(Val(1), Val(2))
TaylorPade2v3 = TaylorPade(Val(2), Val(3))
estimate_order(TaylorPade0v1, prob)
estimate_order(TaylorPade1v2, prob)
estimate_order(TaylorPade2v3, prob)

# sub-sub-diagonal: Taylor-Lobatto solvers
TaylorPade0v2 = TaylorPade(Val(0), Val(2))
TaylorPade1v3 = TaylorPade(Val(1), Val(3))
estimate_order(TaylorPade0v2, prob)
estimate_order(TaylorPade1v3, prob)
