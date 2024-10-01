import ChainRulesCore: rrule, RuleConfig, ProjectTo, backing, @opt_out
using Base.Broadcast: broadcasted

function rrule(::Type{TaylorScalar{T, N}}, v::T, p::NTuple{N, T}) where {N, T}
    taylor_scalar_pullback(t̄) = NoTangent(), value(t̄), partials(t̄)
    return TaylorScalar{T, N}(v, p), taylor_scalar_pullback
end

function rrule(::typeof(value), t::TaylorScalar{T, N}) where {N, T}
    value_pullback(v̄::T) = NoTangent(), TaylorScalar{T, N}(v̄)
    return value(t), value_pullback
end

function rrule(::typeof(partials), t::TaylorScalar{T, N}) where {N, T}
    value_pullback(v̄::NTuple{N, T}) = NoTangent(), TaylorScalar(0, v̄)
    # for structural tangent, convert to tuple
    function value_pullback(v̄::Tangent{P, NTuple{N, T}}) where {P}
        NoTangent(), TaylorScalar{T, N}(zero(T), backing(v̄))
    end
    value_pullback(v̄) = NoTangent(), TaylorScalar{T, N}(zero(T), map(x -> convert(T, x), Tuple(v̄)))
    return partials(t), value_pullback
end

function rrule(::typeof(extract_derivative), t::TaylorScalar{T, N},
        i::Integer) where {N, T}
    function extract_derivative_pullback(d̄)
        NoTangent(), TaylorScalar{T, N}(zero(T), ntuple(j -> j === i ? d̄ : zero(T), Val(N))),
        NoTangent()
    end
    return extract_derivative(t, i), extract_derivative_pullback
end

function rrule(::typeof(*), A::AbstractMatrix{S},
        t::AbstractVector{TaylorScalar{T, N}}) where {N, S <: Real, T <: Real}
    project_A = ProjectTo(A)
    function gemv_pullback(x̄)
        x̂ = reinterpret(reshape, T, x̄)
        t̂ = reinterpret(reshape, T, t)
        NoTangent(), @thunk(project_A(transpose(x̂) * t̂)), @thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

function rrule(::typeof(*), A::AbstractMatrix{S},
        B::AbstractMatrix{TaylorScalar{T, N}}) where {N, S <: Real, T <: Real}
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function gemm_pullback(x̄)
        X̄ = unthunk(x̄)
        NoTangent(),
        @thunk(project_A(X̄ * transpose(B))),
        @thunk(project_B(transpose(A) * X̄))
    end
    return A * B, gemm_pullback
end

(project::ProjectTo{T})(dx::TaylorScalar{T, N}) where {N, T <: Number} = value(dx)

# opt-outs

# Unary functions

for f in (
    exp, exp10, exp2, expm1,
    sin, cos, tan, sec, csc, cot,
    sinh, cosh, tanh, sech, csch, coth,
    log, log10, log2, log1p,
    asin, acos, atan, asec, acsc, acot,
    asinh, acosh, atanh, asech, acsch, acoth,
    sqrt, cbrt, inv
)
    @eval @opt_out frule(::typeof($f), x::TaylorScalar)
    @eval @opt_out rrule(::typeof($f), x::TaylorScalar)
end

# Binary functions

for f in (
    *, /, ^
)
    for (tlhs, trhs) in (
        (TaylorScalar, TaylorScalar),
        (TaylorScalar, Number),
        (Number, TaylorScalar)
    )
        @eval @opt_out frule(::typeof($f), x::$tlhs, y::$trhs)
        @eval @opt_out rrule(::typeof($f), x::$tlhs, y::$trhs)
    end
end

# Multi-argument functions

@opt_out frule(::typeof(*), x::TaylorScalar, y::TaylorScalar, z::TaylorScalar)
@opt_out rrule(::typeof(*), x::TaylorScalar, y::TaylorScalar, z::TaylorScalar)

@opt_out frule(
    ::typeof(*), x::TaylorScalar, y::TaylorScalar, z::TaylorScalar, more::TaylorScalar...)
@opt_out rrule(
    ::typeof(*), x::TaylorScalar, y::TaylorScalar, z::TaylorScalar, more::TaylorScalar...)
