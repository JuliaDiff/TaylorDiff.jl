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

- `value::T`: zeroth order coefficient
- `partials::NTuple{N, T}`: i-th element of this stores the i-th derivative
"""
struct TaylorScalar{T, N} <: Real
    value::T
    partials::NTuple{N, T}
    function TaylorScalar{T, N}(value::T, partials::NTuple{N, T}) where {T, N}
        can_taylorize(T) || throw_cannot_taylorize(T)
        new{T, N}(value, partials)
    end
end

function TaylorScalar(value::T, partials::NTuple{N, T}) where {T, N}
    TaylorScalar{T, N}(value, partials)
end

function TaylorScalar(value_and_partials::NTuple{N, T}) where {T, N}
    TaylorScalar(value_and_partials[1], value_and_partials[2:end])
end

"""
    TaylorScalar{T, N}(x::T) where {T, N}

Construct a Taylor polynomial with zeroth order coefficient.
"""
TaylorScalar{T, N}(x::S) where {T, S <: Real, N} = TaylorScalar(
    T(x), ntuple(i -> zero(T), Val(N)))

"""
    TaylorScalar{T, N}(x::T, d::T) where {T, N}

Construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.
"""
TaylorScalar{T, N}(x::S, d::S) where {T, S <: Real, N} = TaylorScalar(
    T(x), ntuple(i -> i == 1 ? T(d) : zero(T), Val(N)))

function TaylorScalar{T, N}(t::TaylorScalar{T, M}) where {T, N, M}
    v = value(t)
    p = partials(t)
    N <= M ? TaylorScalar(v, p[1:N]) :
    TaylorScalar(v, ntuple(i -> i <= M ? p[i] : zero(T), Val(N)))
end

@inline value(t::TaylorScalar) = t.value
@inline partials(t::TaylorScalar) = t.partials
@inline extract_derivative(t::TaylorScalar, i::Integer) = t.partials[i]
@inline function extract_derivative(v::AbstractArray{T},
        i::Integer) where {T <: TaylorScalar}
    map(t -> extract_derivative(t, i), v)
end
@inline extract_derivative(r, i::Integer) = false
@inline function extract_derivative!(result::AbstractArray, v::AbstractArray{T},
        i::Integer) where {T <: TaylorScalar}
    map!(t -> extract_derivative(t, i), result, v)
end

@inline flatten(t::TaylorScalar) = (value(t), partials(t)...)

function promote_rule(::Type{TaylorScalar{T, N}},
        ::Type{S}) where {T, S, N}
    TaylorScalar{promote_type(T, S), N}
end

function (::Type{F})(x::TaylorScalar{T, N}) where {T, N, F <: AbstractFloat}
    F(value(x))
end

const COVARIANT_OPS = Symbol[:nextfloat, :prevfloat]

for op in COVARIANT_OPS
    @eval Base.$(op)(x::TaylorScalar{T, N}) where {T, N} = TaylorScalar($(op)(value(x)), partials(x))
end

const UNARY_PREDICATES = Symbol[
    :isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(x::TaylorScalar) = $(pred)(value(x))
end
