import Base: convert, promote_rule

export TaylorScalar

"""
    TaylorScalar{T, N}

Representation of Taylor polynomials.

# Fields

- `value::NTuple{N, T}`: i-th element of this stores the (i-1)-th derivative
"""
struct TaylorScalar{T, N}
    value::NTuple{N, T}
end

TaylorOrNumber = Union{TaylorScalar, Number}

@inline TaylorScalar(xs::Vararg{T, N}) where {T, N} = TaylorScalar(xs)

"""
    TaylorScalar{T, N}(x::T) where {T, N}

Construct a Taylor polynomial with zeroth order coefficient.
"""
@generated function TaylorScalar{T, N}(x::T) where {T, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((T(x), $(zeros(T, N - 1)...)))
    end
end

"""
    TaylorScalar{T, N}(x::T, d::T) where {T, N}

Construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.
"""
@generated function TaylorScalar{T, N}(x::T, d::T) where {T, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((T(x), T(d), $(zeros(T, N - 2)...)))
    end
end

@generated function TaylorScalar{T, N}(t::TaylorScalar{T, M}) where {T, N, M}
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

function promote_rule(::Type{TaylorScalar{T, N}},
                      ::Type{S}) where {T, S, N}
    TaylorScalar{promote_type(T, S), N}
end
function promote_rule(::Type{TaylorScalar{T1, N}},
                      ::Type{TaylorScalar{T2, N}}) where {T1, T2, N}
    TaylorScalar{promote_type(T1, T2), N}
end

# Number-like convention (I patched them after removing <: Number)

convert(::Type{TaylorScalar{T, N}}, x::TaylorScalar{T, N}) where {T, N} = x
function convert(::Type{TaylorScalar{T, N}}, x::S) where {T, S, N}
    TaylorScalar{T, N}(convert(T, x))
end
function convert(::Type{TaylorScalar{T1, N}}, x::TaylorScalar{T2, N}) where {T1, T2, N}
    TaylorScalar{T1, N}(convert.(T1, value(x)))
end
