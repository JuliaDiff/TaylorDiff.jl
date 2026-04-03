using TaylorDiff: TaylorDiff, TaylorScalar, make_seed, flatten, get_coefficient,
    set_coefficient, append_coefficient
using OrdinaryDiffEq, Symbolics
using SciMLBase: unwrapped_f

# There are two ways to compute the Taylor coefficients of a ODE solution
# 1. Using naive repeated differentiation
# 2. Using Symbolics to fuse various orders of differentiation together

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
    return u
end

function jetcoeffs!(u, fu, f::ODEFunction{true}, u0, p, t0, ::Val{P}) where {P}
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
    return u
end

"""
# The second method

For ODE u' = f(u, p, t) and initial condition (u0, t0), symbolically computes Taylor expansion of the solution `u` up to order `P`, and then builds a function to evaluate the expression.
"""
function build_jet(f::ODEFunction{iip}, p, order, length = nothing) where {iip}
    f = unwrapped_f(f)
    return build_jet(f, Val{iip}(), p, order, length)
end

function build_jet(f, ::Val{iip}, p, order::Val{P}, length = nothing) where {P, iip}
    if haskey(JET_CACHE, f)
        list = JET_CACHE[f]
        index = findfirst(x -> x[1] == order && x[2] == p, list)
        index !== nothing && return list[index][3]
    end
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

    # Flatten TaylorScalar coefficients for build_function (it doesn't handle custom structs)
    # Then wrap result back into TaylorScalar
    if u isa TaylorScalar
        # Scalar case: build function for array of coefficients
        coeffs = collect(TaylorDiff.flatten(u))
        jet_coeffs = build_function(coeffs, u0, t0; expression = Val(false), cse = true)
        # Wrap to return TaylorScalar
        jet = (u0_val, t0_val) -> TaylorScalar(Tuple(jet_coeffs[1](u0_val, t0_val)))
    elseif u isa AbstractArray && eltype(u) <: TaylorScalar
        # Array case: build function for matrix of coefficients
        # Each row is the coefficients of one TaylorScalar
        n = Base.length(u)
        coeffs_matrix = [TaylorDiff.flatten(u[i])[j] for i in 1:n, j in 1:(P + 1)]
        jet_coeffs = build_function(coeffs_matrix, u0, t0; expression = Val(false), cse = true)
        # Wrap to return array of TaylorScalars
        jet = (
            (u0_val, t0_val) -> begin
                coeffs_out = jet_coeffs[1](u0_val, t0_val)
                return [TaylorScalar(Tuple(coeffs_out[i, :])) for i in 1:n]
            end,
            (out, u0_val, t0_val) -> begin
                coeffs_out = jet_coeffs[2](
                    similar(coeffs_matrix, eltype(u0_val)), u0_val, t0_val
                )
                for i in 1:n
                    out[i] = TaylorScalar(Tuple(coeffs_out[i, :]))
                end
                return out
            end,
        )
    else
        # Fallback (shouldn't happen normally)
        jet = build_function(u, u0, t0; expression = Val(false), cse = true)
    end

    if !haskey(JET_CACHE, f)
        JET_CACHE[f] = []
    end
    push!(JET_CACHE[f], (order, p, jet))
    return jet
end

function build_propagator(f::ODEFunction{iip}, p, coeffs::NTuple{P, Float64}, length = nothing) where {P, iip}
    f = unwrapped_f(f)
    return build_propagator(f, Val{iip}(), p, coeffs, length)
end

function build_propagator(f, ::Val{iip}, p, coeffs::NTuple{P, Float64}, length = nothing) where {P, iip}
    @variables t0::Real dt::Real
    u0 = isnothing(length) ? Symbolics.variable(:u0) : Symbolics.variables(:u0, 1:length)
    if iip
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
    ut = eval_taylor_polynomial(u, (1., coeffs...), dt)
    propagator = build_function(ut, u0, t0, dt; expression = Val(false), cse = true)
    jacobian = Symbolics.jacobian(ut, u0)
    d_propagator = build_function(jacobian, u0, t0, dt; expression = Val(false), cse = true)
    return propagator, d_propagator
end

# Evaluate polynomial for scalar TaylorScalar (returns scalar)
@inline eval_taylor_polynomial(u::TaylorScalar, coeffs, dt) = evalpoly(dt, map(*, coeffs, TaylorDiff.flatten(u)))
# Evaluate polynomial for array of TaylorScalars (returns array)
@inline eval_taylor_polynomial(us::AbstractArray, coeffs, dt) = map(x -> eval_taylor_polynomial(x, coeffs, dt), us)
