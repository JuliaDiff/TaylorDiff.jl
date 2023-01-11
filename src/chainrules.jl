import ChainRulesCore: rrule, RuleConfig
using ZygoteRules: @adjoint

contract(a::TaylorScalar{T, N}, b::TaylorScalar{S, N}) where {T, S, N} = mapreduce(*, +, value(a), value(b))

NONLINEAR_UNARY_FUNCTIONS = Function[
    exp, exp2, exp10, expm1,
    log, log2, log10, log1p,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
]

for func in NONLINEAR_UNARY_FUNCTIONS
    @eval @opt_out rrule(::typeof($func), ::TaylorScalar)
end

NONLINEAR_BINARY_FUNCTIONS = Function[
    *, /, ^
]

for func in NONLINEAR_BINARY_FUNCTIONS
    @eval @opt_out rrule(::typeof($func), ::TaylorScalar, ::TaylorScalar)
    @eval @opt_out rrule(::typeof($func), ::TaylorScalar, ::Number)
    @eval @opt_out rrule(::typeof($func), ::Number, ::TaylorScalar)
end

# Other special cases

@opt_out rrule(::typeof(Base.literal_pow), ::typeof(^), x::TaylorScalar, ::Val{p}) where {p}
@opt_out rrule(::RuleConfig, ::typeof(Base.literal_pow), ::typeof(^), x::TaylorScalar,
               ::Val{p}) where {p}

function rrule(::Type{TaylorScalar{T, N}}, v::NTuple{N, T}) where {N, T <: Number}
    taylor_scalar_pullback(t̄) = NoTangent(), value(t̄)
    return TaylorScalar(v), taylor_scalar_pullback
end

function rrule(::typeof(value), t::TaylorScalar)
    value_pullback(v̄::NTuple) = NoTangent(), TaylorScalar(v̄)
    # for structural tangent, convert to tuple
    value_pullback(v̄) = NoTangent(), TaylorScalar(Tuple(v̄))
    return value(t), value_pullback
end

function rrule(::typeof(extract_derivative), t::TaylorScalar{T, N},
               i::Integer) where {N, T <: Number}
    function extract_derivative_pullback(d̄)
        NoTangent(), TaylorScalar((zeros(T, i - 1)..., d̄, zeros(T, N - i)...)), NoTangent()
    end
    return extract_derivative(t, i), extract_derivative_pullback
end

function rrule(::typeof(*), A::Matrix{T}, t::Vector{TaylorScalar{T, N}}) where {N, T <: Number}
    gemv_pullback(x̄) = NoTangent(), contract.(x̄, transpose(t)), transpose(A) * x̄
    return A * t, gemv_pullback
end

function rrule(::typeof(+), v::Vector{T}, t::Vector{TaylorScalar{T, N}}) where {N, T <: Number}
    vadd_pullback(x̄) = NoTangent(), map(primal, x̄), x̄
    return v + t, vadd_pullback
end

function rrule(::typeof(+), t::Vector{TaylorScalar{T, N}}, v::Vector{T}) where {N, T <: Number}
    vadd_pullback(x̄) = NoTangent(), x̄, map(primal, x̄)
    return t + v, vadd_pullback
end

@adjoint +(t::Vector{TaylorScalar{T, N}}, v::Vector{T}) where {N, T <: Number} = t + v, x̄ -> (x̄, map(primal, x̄))

@adjoint +(v::Vector{T}, t::Vector{TaylorScalar{T, N}}) where {N, T <: Number} = v + t, x̄ -> (map(primal, x̄), x̄)
