using TaylorDiff: TaylorDiff, TaylorScalar, make_seed, flatten, get_coefficient,
                  set_coefficient, append_coefficient
using TaylorSeries, TaylorIntegration
using LinearAlgebra
using ODEProblemLibrary, OrdinaryDiffEq, BenchmarkTools, Symbolics
using CairoMakie

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

function jetcoeffs_inplace!(u, fu, f::ODEFunction{true}, u0, p, t0, ::Val{P}) where {P}
    t = TaylorScalar{P}(t0, one(t0))
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
    u_term = make_term.(u)
    build_function(u_term, u0, t0; expression = Val(false), cse = true)
end

function make_term(a)
    term(TaylorScalar, Symbolics.unwrap(a.value), map(Symbolics.unwrap, a.partials))
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

function plot_simplification_effect()
    prob = ODEProblem(pcr3bp!, q0, tspan, p)
    t0 = prob.tspan[1]
    raw = Float64[]
    optimized = Float64[]
    orders = [4, 6, 8, 10, 12]
    for P in orders
        raw_time = @belapsed jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
        fast_oop, fast_iip = build_jetcoeffs(prob.f, prob.p, Val(P), length(prob.u0))
        optimized_time = @belapsed $fast_oop($prob.u0, $t0)
        push!(raw, raw_time)
        push!(optimized, optimized_time)
    end

    colors = Makie.wong_colors()
    f = begin
        f = Figure(resolution = (700, 400))
        ax = Axis(f[1, 1],
            xlabel = "Order",
            ylabel = "Time for computing Taylor polynomial (s)",
            title = "Effect of Symbolic Simplification on Computing Taylor Polynomial",
            xticks = orders,
            yscale = log10
        )

        group = [fill(1, length(orders));
                 fill(2, length(orders))]
        barplot!(ax, [orders; orders], [raw; optimized],
            dodge = group,
            color = colors[group])
        elements = [PolyElement(polycolor = colors[i]) for i in 1:2]
        Legend(f[1, 2], elements, ["Naive", "Simplified"], "Groups")
        f
    end
    save("simplification_effect.png", f)
end

plot_simplification_effect()

@taylorize function pcr3bp!(dq, q, param, t)
    local μ = param[1]
    local onemμ = 1 - μ
    x1 = q[1]-μ
    x1sq = x1^2
    y = q[2]
    ysq = y^2
    r1_1p5 = (x1sq+ysq)^1.5
    x2 = q[1]+onemμ
    x2sq = x2^2
    r2_1p5 = (x2sq+ysq)^1.5
    dq[1] = q[3] + q[2]
    dq[2] = q[4] - q[1]
    dq[3] = (-((onemμ*x1)/r1_1p5) - ((μ*x2)/r2_1p5)) + q[4]
    dq[4] = (-((onemμ*y )/r1_1p5) - ((μ*y )/r2_1p5)) - q[3]
    return nothing
end

V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
H(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)
H(x) = H(x...)
t0 = 0.0
μ = 0.01
J0 = -1.58
function py!(q0, J0)
    @assert iszero(q0[2]) && iszero(q0[3]) # q0[2] and q0[3] have to be equal to zero
    q0[4] = q0[1] + sqrt( q0[1]^2-2( V(q0[1], q0[2])-J0 ) )
    nothing
end
q0 = [-0.8, 0.0, 0.0, 0.0]
py!(q0, J0)
q0
tspan = (0.0, 2000.0)
p = [μ]
prob = ODEProblem(pcr3bp!, q0, tspan, p)
raw = Float64[]
optimized = Float64[]
orders = [4, 6, 8, 10, 12]
ti_raw = Float64[]
ti_optimized = Float64[]
for P in orders
    raw_time = @belapsed jetcoeffs($prob.f, $prob.u0, $prob.p, $t0, Val($P))
    fast_oop, fast_iip = build_jetcoeffs(prob.f, prob.p, Val(P), length(prob.u0))
    optimized_time = @belapsed $fast_oop($prob.u0, $t0)
    push!(raw, raw_time)
    push!(optimized, optimized_time)
    # TI
    tt = Taylor1(0.0, P)
    q = [Taylor1(x, P) for x in q0]
    dq = [Taylor1(0.0, P) for x in q0]
    qaux = [Taylor1(0.0, P) for x in q0]
    a = TaylorIntegration._allocate_jetcoeffs!(Val(pcr3bp!), tt, q, dq, p)
    ti_time = @belapsed TaylorIntegration.jetcoeffs!($pcr3bp!, $tt, $q, $dq, $qaux, $p)
    ti_optimized_time = @belapsed TaylorIntegration.jetcoeffs!(Val(pcr3bp!), $tt, $q, $dq, $p, $a)
    push!(ti_raw, ti_time)
    push!(ti_optimized, ti_optimized_time)
end

colors = Makie.wong_colors()
f = begin
    f = Figure(resolution = (700, 400))
    ax = Axis(f[1, 1],
        xlabel = "Order",
        ylabel = "Time for computing Taylor polynomial (s)",
        title = "Effect of Symbolic Simplification on Computing Taylor Polynomial",
        xticks = orders,
        yscale = log10
    )

    group = [fill(1, length(orders));
                fill(2, length(orders));
                fill(3, length(orders));
                fill(4, length(orders))]
    barplot!(ax, [orders; orders; orders; orders], [raw; optimized; ti_raw; ti_optimized],
        dodge = group,
        color = colors[group])
    elements = [PolyElement(polycolor = colors[i]) for i in 1:4]
    Legend(f[1, 2], elements, ["Naive", "Simplified", "TI Naive", "TI Simplified"], "Groups")
    f
end
save("simplification_with_ti.png", f)
