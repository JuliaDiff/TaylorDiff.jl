import Base: zero, one, adjoint, conj, transpose
import Base: +, -, *, /
import Base: convert, promote_rule

export TaylorScalar

"""
    TaylorDiff.can_taylor(V::Type)

Determines whether the type V is allowed as the scalar type in a
Dual. By default, only `<:Real` types are allowed.
"""
can_taylorize(::Type{<:Real}) = true
can_taylorize(::Type) = false

@noinline function throw_cannot_taylorize(V::Type)
    throw(ArgumentError("Cannot create a Taylor polynomial over scalar type $V." *
                        " If the type behaves as a scalar, define TaylorDiff.can_taylorize(::Type{$V}) = true."))
end

"""
    TaylorScalar{T, N}

Representation of Taylor polynomials.

# Fields

- `value::NTuple{N, T}`: i-th element of this stores the (i-1)-th derivative
"""
struct TaylorScalar{T, N} <: Real
    value::NTuple{N, T}
    function TaylorScalar{T, N}(value::NTuple{N, T}) where {T, N}
        can_taylorize(T) || throw_cannot_taylorize(T)
        new{T, N}(value)
    end
end

TaylorScalar(value::NTuple{N, T}) where {T, N} = TaylorScalar{T, N}(value)
TaylorScalar(value::Vararg{T, N}) where {T, N} = TaylorScalar{T, N}(value)

"""
    TaylorScalar{T, N}(x::T) where {T, N}

Construct a Taylor polynomial with zeroth order coefficient.
"""
@generated function TaylorScalar{T, N}(x::S) where {T, S <: Real, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorScalar((T(x), $(zeros(T, N - 1)...)))
    end
end

"""
    TaylorScalar{T, N}(x::T, d::T) where {T, N}

Construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.
"""
@generated function TaylorScalar{T, N}(x::S, d::S) where {T, S <: Real, N}
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
@inline function extract_derivative(v::AbstractArray{T},
        i::Integer) where {T <: TaylorScalar}
    map(t -> extract_derivative(t, i), v)
end
@inline extract_derivative(r, i::Integer) = false
@inline function extract_derivative!(result::AbstractArray, v::AbstractArray{T},
        i::Integer) where {T <: TaylorScalar}
    map!(t -> extract_derivative(t, i), result, v)
end
@inline primal(t::TaylorScalar) = extract_derivative(t, 1)

function promote_rule(::Type{TaylorScalar{T, N}},
        ::Type{S}) where {T, S, N}
    TaylorScalar{promote_type(T, S), N}
end
function promote_rule(::Type{TaylorScalar{T1, N}}, ::Type{TaylorScalar{T2,N}}) where {T1, T2, N}
TaylorScalar{promote_type(T1,T2), N}
end

function (::Type{F})(x::TaylorScalar{T, N}) where {T, N, F <: AbstractFloat}
    F(primal(x))
end

function Base.nextfloat(x::TaylorScalar{T, N}) where {T, N}
    TaylorScalar{T, N}(ntuple(i -> i == 1 ? nextfloat(value(x)[i]) : value(x)[i], N))
end

function Base.prevfloat(x::TaylorScalar{T, N}) where {T, N}
    TaylorScalar{T, N}(ntuple(i -> i == 1 ? prevfloat(value(x)[i]) : value(x)[i], N))
end

const UNARY_PREDICATES = Symbol[
    :isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(x::TaylorScalar) = $(pred)(primal(x))
end
