include("integration.jl")

struct ImplicitFirstOrder{T}
    μ::T
end

function (solver::ImplicitFirstOrder)(prob; h)
    μ = solver.μ
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(1), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            du = get_coefficient(fast_oop(u_next, t + μ * h), 1)
            u_next .- u_curr .- μ * h .* du
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        f = get_coefficient(fast_oop(sol.u, t + μ * h), 1)
        u .= sol.u + (1 - μ) * h * f
        t += h
    end
    return u
end

struct ImplicitSecondOrder{T}
    μ::T
end

function (solver::ImplicitSecondOrder)(prob; h)
    μ = solver.μ
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(2), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + μ * h)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u_curr + μ * h * u1 - (μ * h)^2 * u2 - u_next
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        poly = fast_oop(sol.u, t + μ * h)
        u1 = get_coefficient(poly, 1)
        u2 = get_coefficient(poly, 2)
        u .= real(sol.u + (1 - μ) * h * u1 + ((1 - μ) * h)^2 * u2)
        t += h
    end
    return u
end

struct ImplicitThirdOrder{T}
    μ::T
end

function (solver::ImplicitThirdOrder)(prob; h)
    μ = solver.μ
    uh = μ * h
    cuh = (1 - μ) * h
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(3), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u_curr + uh * u1 - uh^2 * u2 + uh^3 * u3 - u_next
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        poly = fast_oop(sol.u, t + uh)
        u1 = get_coefficient(poly, 1)
        u2 = get_coefficient(poly, 2)
        u3 = get_coefficient(poly, 3)
        u .= sol.u + cuh * u1 + cuh^2 * u2 + cuh^3 * u3
        t += h
    end
    return u
end

struct ImplicitThirdOrderExtrapolated{T1, T2}
    μ1::T1
    c1::T1
    μ2::T2
    c2::T1
end

function (solver::ImplicitThirdOrderExtrapolated)(prob; h)
    uh1 = solver.μ1 * h
    cuh1 = (1 - solver.μ1) * h
    uh2 = solver.μ2 * h
    cuh2 = (1 - solver.μ2) * h
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(3), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh1)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u_curr + uh1 * u1 - uh1^2 * u2 + uh1^3 * u3 - u_next
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        poly = fast_oop(sol.u, t + uh1)
        u1 = get_coefficient(poly, 1)
        u2 = get_coefficient(poly, 2)
        u3 = get_coefficient(poly, 3)
        u_trial1 = real(sol.u + cuh1 * u1 + cuh1^2 * u2 + cuh1^3 * u3)
        nlf2 = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh2)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u_curr + uh2 * u1 - uh2^2 * u2 + uh2^3 * u3 - u_next
        end
        guess2 = copy(u)
        nlprob2 = NonlinearProblem(nlf2, guess2, u)
        sol2 = solve(nlprob2)
        poly2 = fast_oop(sol2.u, t + uh2)
        u1_2 = get_coefficient(poly2, 1)
        u2_2 = get_coefficient(poly2, 2)
        u3_2 = get_coefficient(poly2, 3)
        u_trial2 = real(sol2.u + cuh2 * u1_2 + cuh2^2 * u2_2 + cuh2^3 * u3_2)
        u .= (0.8 * u_trial1 + 0.2 * u_trial2)
        t += h
    end
    return u
end

struct ImplicitFourthOrder{T}
    μ::T
end

function (solver::ImplicitFourthOrder)(prob; h)
    μ = solver.μ
    uh = μ * h
    cuh = (1 - μ) * h
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(4), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u4 = get_coefficient(poly, 4)
            u_curr + uh * u1 - uh^2 * u2 + uh^3 * u3 - uh^4 * u4 - u_next
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        poly = fast_oop(sol.u, t + uh)
        u1 = get_coefficient(poly, 1)
        u2 = get_coefficient(poly, 2)
        u3 = get_coefficient(poly, 3)
        u4 = get_coefficient(poly, 4)
        u .= sol.u + cuh * u1 + cuh^2 * u2 + cuh^3 * u3 + cuh^4 * u4
        t += h
    end
    return u
end

struct ImplicitFourthOrderExtrapolated{T1, T2}
    μ1::T2
    c1::T1
    μ2::T2
    c2::T1
end

function (solver::ImplicitFourthOrderExtrapolated)(prob; h)
    uh1 = solver.μ1 * h
    cuh1 = (1 - solver.μ1) * h
    uh2 = solver.μ2 * h
    cuh2 = (1 - solver.μ2) * h
    fast_oop, _ = build_jetcoeffs(prob.f, prob.p, Val(4), length(prob.u0))
    t0, t_end = prob.tspan
    t = t0
    u = copy(prob.u0)
    while t < t_end
        nlf = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh1)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u4 = get_coefficient(poly, 4)
            u_curr + uh1 * u1 - uh1^2 * u2 + uh1^3 * u3 - uh1^4 * u4 - u_next
        end
        guess = copy(u)
        nlprob = NonlinearProblem(nlf, guess, u)
        sol = solve(nlprob)
        poly = fast_oop(sol.u, t + uh1)
        u1 = get_coefficient(poly, 1)
        u2 = get_coefficient(poly, 2)
        u3 = get_coefficient(poly, 3)
        u4 = get_coefficient(poly, 4)
        u_trial1 = real(sol.u + cuh1 * u1 + cuh1^2 * u2 + cuh1^3 * u3 + cuh1^4 * u4)
        nlf2 = (u_next, u_curr) -> begin
            poly = fast_oop(u_next, t + uh2)
            u1 = get_coefficient(poly, 1)
            u2 = get_coefficient(poly, 2)
            u3 = get_coefficient(poly, 3)
            u4 = get_coefficient(poly, 4)
            u_curr + uh2 * u1 - uh2^2 * u2 + uh2^3 * u3 - uh2^4 * u4 - u_next
        end
        guess2 = copy(u)
        nlprob2 = NonlinearProblem(nlf2, guess2, u)
        sol2 = solve(nlprob2)
        poly2 = fast_oop(sol2.u, t + uh2)
        u1_2 = get_coefficient(poly2, 1)
        u2_2 = get_coefficient(poly2, 2)
        u3_2 = get_coefficient(poly2, 3)
        u4_2 = get_coefficient(poly2, 4)
        u_trial2 = real(sol2.u + cuh2 * u1_2 + cuh2^2 * u2_2 + cuh2^3 * u3_2 +
                        cuh2^4 * u4_2)
        u .= (solver.c1 * u_trial1 + solver.c2 * u_trial2) / (solver.c1 + solver.c2)
        t += h
    end
    return u
end

function estimate_order(solver, prob; dt1 = 0.01, dt2 = 0.005)
    ref = prob.f.analytic(prob.u0, prob.p, prob.tspan[2])
    println("Reference: ", ref)
    u1 = solver(prob; h = dt1)
    println("u1: ", u1)
    u2 = solver(prob; h = dt2)
    println("u2: ", u2)
    err1 = norm(u1 - ref)
    err2 = norm(u2 - ref)
    order = log(err1 / err2) / log(dt1 / dt2)
    return order
end

init = rand(2)
linear_f = ODEFunction(
    ODEProblemLibrary.f_2dlinear, analytic = ODEProblemLibrary.f_2dlinear_analytic)
prob = ODEProblem(linear_f, init, (0.0, 1.0), 1.01)
prob_complex = ODEProblem(
    linear_f, complex(init), (0.0, 1.0), 1.01)

ImplicitEuler = ImplicitFirstOrder(1.0)
ImplicitMidpoint = ImplicitFirstOrder(0.5)
estimate_order(ImplicitEuler, prob)
estimate_order(ImplicitMidpoint, prob)

ImplicitTaylor2 = ImplicitSecondOrder(1.0)
ImplicitTaylor2Midpoint = ImplicitSecondOrder(0.5)
ImplicitTaylor2Complex = ImplicitSecondOrder(0.5 + sqrt(3) * im / 6)
estimate_order(ImplicitTaylor2, prob)
estimate_order(ImplicitTaylor2Midpoint, prob)
estimate_order(ImplicitTaylor2Complex, prob_complex)

ImplicitTaylor3Midpoint = ImplicitThirdOrder(0.5)
ImplicitTaylor3Extrapolated = ImplicitThirdOrderExtrapolated(0.5, 0.8, 0.5 + 0.5im, 0.2)
estimate_order(ImplicitTaylor3Midpoint, prob)
estimate_order(ImplicitTaylor3Extrapolated, prob_complex; dt1 = 1 / 12, dt2 = 1 / 16)

ImplicitTaylor4Midpoint = ImplicitFourthOrder(0.5)
ImplicitTaylor4Extrapolated = ImplicitFourthOrderExtrapolated(
    0.5 + 0.5im * tan(π / 10), tan(π / 10) * sec(π / 10)^5,
    0.5 + 0.5im * tan(3π / 10), tan(3π / 10) * sec(3π / 10)^5)
estimate_order(ImplicitTaylor4Midpoint, prob)
estimate_order(ImplicitTaylor4Extrapolated, prob_complex; dt1 = 1 / 9, dt2 = 1 / 12)
