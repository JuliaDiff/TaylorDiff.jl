export TaylorScalar

"""
    TaylorScalar{T, P}

Representation of Taylor polynomials.

# Fields

- `value::T`: zeroth order coefficient
- `partials::NTuple{P, T}`: i-th element of this stores the i-th derivative
"""
struct TaylorScalar{T, P} <: Real
    value::T
    partials::NTuple{P, T}
    function TaylorScalar(value::T, partials::NTuple{P, T}) where {T, P}
        can_taylorize(T) || throw_cannot_taylorize(T)
        new{T, P}(value, partials)
    end
end

# Allowing promotion of basic Number types
TaylorScalar{T, P}(x) where {T, P} = TaylorScalar{P}(T(x))

# Allowing construction with flattened value and partials in a tuple
TaylorScalar(all::NTuple{P, T}) where {T, P} = TaylorScalar(all[1], all[2:end])

"""
    TaylorScalar{P}(value::T) where {T, P}

Convenience function: construct a Taylor polynomial with zeroth order coefficient.
"""
TaylorScalar{P}(value::T) where {T, P} = TaylorScalar(value, ntuple(i -> zero(T), Val(P)))

"""
    TaylorScalar{P}(value::T, seed::T)

Convenience function: construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.
"""
TaylorScalar{P}(value::T, seed::T) where {T, P} = TaylorScalar(
    value, ntuple(i -> i == 1 ? seed : zero(T), Val(P)))

# Truncate or extend the order of a Taylor polynomial.
function TaylorScalar{P}(t::TaylorScalar{T, Q}) where {T, P, Q}
    v = value(t)
    p = partials(t)
    P <= Q ? TaylorScalar(v, p[1:P]) :
    TaylorScalar(v, ntuple(i -> i <= Q ? p[i] : zero(T), Val(P)))
end

# Covariant: operate on the value, and reconstruct with the partials
for op in Symbol[:nextfloat, :prevfloat]
    @eval Base.$(op)(x::TaylorScalar) = TaylorScalar(Base.$(op)(value(x)), partials(x))
end

# Invariant: operate on the value, and drop the partials
for op in Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]
    @eval Base.$(op)(x::TaylorScalar) = Base.$(op)(value(x))
end
