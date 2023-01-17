import ChainRulesCore: rrule, RuleConfig, ProjectTo, backing
using ZygoteRules: @adjoint

function contract(a::TaylorScalar{T, N}, b::TaylorScalar{S, N}) where {T, S, N}
    mapreduce(*, +, value(a), value(b))
end

NONLINEAR_UNARY_FUNCTIONS = Function[exp, exp2, exp10, expm1,
                                     log, log2, log10, log1p,
                                     inv, sqrt, cbrt,
                                     sin, cos, tan, cot, sec, csc,
                                     asin, acos, atan, acot, asec, acsc,
                                     sinh, cosh, tanh, coth, sech, csch,
                                     asinh, acosh, atanh, acoth, asech, acsch]

for func in NONLINEAR_UNARY_FUNCTIONS
    @eval @opt_out rrule(::typeof($func), ::TaylorScalar)
end

NONLINEAR_BINARY_FUNCTIONS = Function[*, /, ^]

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

function rrule(::typeof(value), t::TaylorScalar{T, N}) where {N, T}
    value_pullback(v̄::NTuple{N, T}) = NoTangent(), TaylorScalar(v̄)
    # for structural tangent, convert to tuple
    value_pullback(v̄::Tangent{P, NTuple{N, T}}) where P = NoTangent(), TaylorScalar{T, N}(backing(v̄))
    value_pullback(v̄) = NoTangent(), TaylorScalar{T, N}(map(x -> convert(T, x), Tuple(v̄)))
    return value(t), value_pullback
end

function rrule(::typeof(extract_derivative), t::TaylorScalar{T, N},
               i::Integer) where {N, T <: Number}
    function extract_derivative_pullback(d̄)
        NoTangent(), TaylorScalar{T, N}(ntuple(j -> j === i ? d̄ : zero(T), Val(N))),
        NoTangent()
    end
    return extract_derivative(t, i), extract_derivative_pullback
end

function rrule(::typeof(*), A::Matrix{S},
               t::Vector{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = ProjectTo(A)
    function gemv_pullback(x̄)
        NoTangent(), @thunk(project_A(contract.(x̄, transpose(t)))), @thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

@adjoint function +(t::Vector{TaylorScalar{T, N}}, v::Vector{T}) where {N, T <: Number}
    project_v = ProjectTo(v)
    t + v, x̄ -> (x̄, project_v(x̄))
end

@adjoint function +(v::Vector{T}, t::Vector{TaylorScalar{T, N}}) where {N, T <: Number}
    project_v = ProjectTo(v)
    v + t, x̄ -> (project_v(x̄), x̄)
end

(project::ProjectTo{T})(dx::TaylorScalar{T, N}) where {N, T <: Number} = primal(dx)
