import ChainRulesCore: rrule

function rrule(::Type{TaylorScalar}, v)
    taylor_scalar_pullback(t̄) = NoTangent(), t̄.value
    return TaylorScalar(v), taylor_scalar_pullback
end

function rrule(::typeof(value), t::TaylorScalar)
    value_pullback(v̄) = NoTangent(), Tangent{TaylorScalar}(; value = v̄)
    return value(t), value_pullback
end

@generated function wrap_derivative(d̄::T, i::Integer, N::Integer) where T <: Number
    return quote
        $(Expr(:meta, :inline))
        Tangent{TaylorScalar}(
            ; value = ($(zeros(T, i - 1)...), d̄, $(zeros(T, N - i)))
        )
    end
end

function rrule(::typeof(extract_derivative), t::NTuple{N, T}, i::Integer) where {N, T <: Number}
    extract_derivative_pullback(d̄) = NoTangent(), wrap_derivative(d̄, i, N), NoTangent()
    return extract_derivative(t, i), extract_derivative_pullback
end
