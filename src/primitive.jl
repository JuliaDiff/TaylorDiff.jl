import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch, sinpi, cospi
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: sinc, cosc, hypot, max, min, literal_pow, sincos

Taylor = Union{TaylorScalar, TaylorArray}

@inline value(t::Taylor) = t.value
@inline partials(t::Taylor) = t.partials
@inline @generated extract_derivative(t::Taylor, ::Val{P}) where {P} = :(t.partials[P] *
                                                                         $(factorial(P)))
@inline extract_derivative(a::AbstractArray{<:TaylorScalar}, p) = map(
    t -> extract_derivative(t, p), a)
@inline extract_derivative(result, p) = zero(result)
@inline extract_derivative!(result, a::AbstractArray{<:TaylorScalar}, p) = map!(
    t -> extract_derivative(t, p), result, a)

@inline flatten(t::Taylor) = (value(t), partials(t)...)

function Base.promote_rule(::Type{TaylorScalar{T, P}},
        ::Type{S}) where {T, S, P}
    TaylorScalar{promote_type(T, S), P}
end

function (::Type{F})(x::TaylorScalar{T, P}) where {T, P, F <: AbstractFloat}
    F(value(x))
end

# Unary

## Delegated

@inline -(t::TaylorScalar) = TaylorScalar(-value(t), .-partials(t))
@inline sqrt(t::TaylorScalar) = t^0.5
@inline cbrt(t::TaylorScalar) = ^(t, 1 / 3)
@inline inv(t::TaylorScalar) = one(t) / t
@inline sinpi(t::TaylorScalar) = sin(π * t)
@inline cospi(t::TaylorScalar) = cos(π * t)
@inline exp10(t::TaylorScalar) = exp(t * log(10))
@inline exp2(t::TaylorScalar) = exp(t * log(2))
@inline expm1(t::TaylorScalar) = TaylorScalar(expm1(value(t)), partials(exp(t)))

## Hand-written exp, sin, cos

@immutable function exp(t::TaylorScalar{T, P}) where {P, T}
    f = flatten(t)
    v[0] = exp(f[0])
    for i in 1:P
        for j in 0:(i - 1)
            v[i] += (i - j) * v[j] * f[i - j]
        end
        v[i] /= i
    end
    return TaylorScalar(v)
end

for func in (:sin, :cos)
    @eval @immutable function $func(t::TaylorScalar{T, P}) where {T, P}
        f = flatten(t)
        s[0], c[0] = sincos(f[0])
        for i in 1:P
            for j in 0:(i - 1)
                s[i] += (i - j) * c[j] * f[i - j]
                c[i] -= (i - j) * s[j] * f[i - j]
            end
            s[i] /= i
            c[i] /= i
        end
        return $(func == :sin ? :(TaylorScalar(s)) : :(TaylorScalar(c)))
    end
end

sincos(t::TaylorScalar) = (sin(t), cos(t))

# Binary

## Easy case

@inline +(a::Number, b::TaylorScalar) = TaylorScalar(a + value(b), partials(b))
@inline -(a::Number, b::TaylorScalar) = TaylorScalar(a - value(b), .-partials(b))
@inline *(a::Number, b::TaylorScalar) = TaylorScalar(a * value(b), a .* partials(b))
@inline /(a::Number, b::TaylorScalar) = /(promote(a, b)...)

@inline +(a::TaylorScalar, b::Number) = TaylorScalar(value(a) + b, partials(a))
@inline -(a::TaylorScalar, b::Number) = TaylorScalar(value(a) - b, partials(a))
@inline *(a::TaylorScalar, b::Number) = TaylorScalar(value(a) * b, partials(a) .* b)
@inline /(a::TaylorScalar, b::Number) = TaylorScalar(value(a) / b, partials(a) ./ b)

const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

for op in [:>, :<, :(==), :(>=), :(<=)]
    for R in AMBIGUOUS_TYPES
        @eval @inline $op(a::TaylorScalar, b::$R) = $op(value(a)[1], b)
        @eval @inline $op(a::$R, b::TaylorScalar) = $op(a, value(b)[1])
    end
    @eval @inline $op(a::TaylorScalar, b::TaylorScalar) = $op(value(a)[1], value(b)[1])
end

@inline +(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(
    value(a) + value(b), map(+, partials(a), partials(b)))
@inline -(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(
    value(a) - value(b), map(-, partials(a), partials(b)))

@immutable function *(a::TaylorScalar{T, P}, b::TaylorScalar{T, P}) where {T, P}
    va, vb = flatten(a), flatten(b)
    for i in 0:P
        for j in 0:i
            v[i] += va[j] * vb[i - j]
        end
    end
    TaylorScalar(v)
end

@immutable function /(a::TaylorScalar{T, P}, b::TaylorScalar{T, P}) where {T, P}
    va, vb = flatten(a), flatten(b)
    v[0] = va[0] / vb[0]
    for i in 1:P
        v[i] = va[i]
        for j in 0:(i - 1)
            v[i] -= vb[i - j] * v[j]
        end
        v[i] /= vb[0]
    end
    TaylorScalar(v)
end

@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{0}) = one(x)
@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{1}) = x
@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{2}) = x * x
@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{3}) = x * x * x
@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{-1}) = inv(x)
@inline literal_pow(::typeof(^), x::TaylorScalar, ::Val{-2}) = (i = inv(x); i * i)

for R in (Integer, Real)
    @eval @immutable function ^(t::TaylorScalar{T, P}, n::S) where {S <: $R, T, P}
        f = flatten(t)
        v[0] = f[0]^n
        for i in 1:P
            for j in 0:(i - 1)
                v[i] += (n * (i - j) - j) * v[j] * f[i - j]
            end
            v[i] /= (i * f[0])
        end
        return TaylorScalar(v)
    end
    @eval ^(a::S, t::TaylorScalar) where {S <: $R} = exp(t * log(a))
end

^(t::TaylorScalar, s::TaylorScalar) = exp(s * log(t))

@inline function differentiate(t::TaylorScalar{T, P}) where {T, P}
    s = partials(t)
    TaylorScalar(ntuple(i -> s[i] * i, Val(P)))
end
@inline function integrate(t::TaylorScalar{T, P}, C::T) where {T, P}
    s = flatten(t)
    TaylorScalar(C, ntuple(i -> s[i] / i, Val(P + 1)))
end
@inline raise(f0, d::TaylorScalar, t) = integrate(differentiate(t) * d, f0)
@inline raise(f0, d::Number, t) = d * t
@inline raiseinv(f0, d, t) = integrate(differentiate(t) / d, f0)

# Array primitives

# Pass-through linear operators

# *(a::AbstractMatrix{T}, b::TaylorArray{T}) where {T} = TaylorArray(a * value(b), map(p -> a * p, partials(b)))
