import Base: zero, one, adjoint, conj
import Base: convert, promote_rule

export TaylorScalar

"""
    TaylorScalar{T <: Number, N}

Representation of Taylor polynomials.

# Fields

- `value::NTuple{N, T}`: i-th element of this stores the (i-1)-th derivative
"""
struct TaylorScalar{T <: Number, N} <: Number
    value::NTuple{N, T}
end

@inline TaylorScalar(xs::Vararg{T, N}) where {T <: Number, N} = TaylorScalar(xs)

"""
    TaylorScalar{T, N}(x::T) where {T <: Number, N}

Construct a seed with unit first-order perturbation.
"""
@generated function TaylorScalar{T, N}(x::T) where {T <: Number, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((x, $(zeros(T, N - 1)...)))
    end
end

"""
    TaylorScalar{T, N}(x::T, d::T) where {T <: Number, N}

Construct a seed with first-order perturbation.
"""
@generated function TaylorScalar{T, N}(x::T, d::T) where {T <: Number, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((x, d, $(zeros(T, N - 2)...)))
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

@generated function zero(::Type{TaylorScalar{T, N}}) where {T, N}
    quote
        $(Expr(:meta, :inline))
        TaylorScalar($(zeros(T, N)...))
    end
end
@generated function one(::Type{TaylorScalar{T, N}}) where {T, N}
    quote
        $(Expr(:meta, :inline))
        TaylorScalar(one(T), $(zeros(T, N - 1)...))
    end
end

@inline zero(::TaylorScalar{T, N}) where {T, N} = zero(TaylorScalar{T, N})
@inline one(::TaylorScalar{T, N}) where {T, N} = one(TaylorScalar{T, N})

adjoint(t::TaylorScalar) = t
conj(t::TaylorScalar) = t

function promote_rule(::Type{TaylorScalar{T, N}},
                      ::Type{S}) where {T <: Number, S <: Number, N}
    TaylorScalar{promote_type(T, S), N}
end
function promote_rule(::Type{TaylorScalar{T, N}},
                      ::Type{TaylorScalar{S, N}}) where {T <: Number, S <: Number, N}
    TaylorScalar{promote_type(T, S), N}
end
