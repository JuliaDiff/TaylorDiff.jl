import Base: zero, one, adjoint, conj, transpose
import Base: +, -, *, /
import Base: convert, promote_rule

export TaylorScalar

"""
    TaylorScalar{T <: Number, N}

Representation of Taylor polynomials.

# Fields

- `value::NTuple{N, T}`: i-th element of this stores the (i-1)-th derivative
"""
struct TaylorScalar{T <: Number, N}
    value::NTuple{N, T}
end

@inline TaylorScalar(xs::Vararg{T, N}) where {T <: Number, N} = TaylorScalar(xs)

"""
    TaylorScalar{T, N}(x::S) where {S <: Number, T <: Number, N}

Construct a Taylor polynomial with zeroth order coefficient.
"""
@generated function TaylorScalar{T, N}(x::S) where {S <: Number, T <: Number, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((T(x), $(zeros(T, N - 1)...)))
    end
end

"""
    TaylorScalar{T, N}(x::S, d::S) where {S <: Number, T <: Number, N}

Construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.
"""
@generated function TaylorScalar{T, N}(x::S, d::S) where {S <: Number, T <: Number, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((T(x), T(d), $(zeros(T, N - 2)...)))
    end
end

@generated function TaylorScalar{T, N}(t::TaylorScalar{T, M}) where {T <: Number, N, M}
    N <= M ? quote
        $(Expr(:meta, :inline))
        TaylorScalar(value(t)[1:N])
    end : quote
        $(Expr(:meta, :inline))
        TaylorScalar((value(t)..., $(zeros(T, N - M)...)))
    end
end

@inline value(t::TaylorScalar) = t.value
@inline extract_derivative(t::TaylorScalar, i::Integer) = t.value[i]
@inline extract_derivative(r, i::Integer) = false
@inline primal(t::TaylorScalar) = extract_derivative(t, 1)

@inline zero(::Type{TaylorScalar{T, N}}) where {T, N} = TaylorScalar{T, N}(zero(T))
@inline one(::Type{TaylorScalar{T, N}}) where {T, N} = TaylorScalar{T, N}(one(T))
@inline zero(::TaylorScalar{T, N}) where {T, N} = zero(TaylorScalar{T, N})
@inline one(::TaylorScalar{T, N}) where {T, N} = one(TaylorScalar{T, N})

adjoint(t::TaylorScalar) = t
conj(t::TaylorScalar) = t

function promote_rule(::Type{TaylorScalar{T, N}},
                      ::Type{S}) where {T <: Number, S <: Number, N}
    TaylorScalar{promote_type(T, S), N}
end

# Number-like convention (I patched them after removing <: Number)

convert(::Type{TaylorScalar{T, N}}, x::Number) where {T, N} = TaylorScalar{T, N}(x)
for op in (:+, :-, :*, :/)
    @eval @inline $op(a::TaylorScalar, b::Number) = $op(promote(a, b)...)
    @eval @inline $op(a::Number, b::TaylorScalar) = $op(promote(a, b)...)
end
transpose(t::TaylorScalar) = t
