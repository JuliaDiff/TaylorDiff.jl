import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch, sinpi, cospi
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch
import Base: sinc, cosc
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min
import Base: tail

# Unary
@inline +(a::Number, b::TaylorScalar) = TaylorScalar(a + value(b), partials(b))
@inline -(a::Number, b::TaylorScalar) = TaylorScalar(a - value(b), map(-, partials(b)))
@inline *(a::Number, b::TaylorScalar) = TaylorScalar(a * value(b), a .* partials(b))
@inline /(a::Number, b::TaylorScalar) = /(promote(a, b)...)

@inline +(a::TaylorScalar, b::Number) = TaylorScalar(value(a) + b, partials(a))
@inline -(a::TaylorScalar, b::Number) = TaylorScalar(value(a) - b, partials(a))
@inline *(a::TaylorScalar, b::Number) = TaylorScalar(value(a) * b, partials(a) .* b)
@inline /(a::TaylorScalar, b::Number) = TaylorScalar(value(a) / b, partials(a) ./ b)

## Delegated

@inline sqrt(t::TaylorScalar) = t^0.5
@inline cbrt(t::TaylorScalar) = ^(t, 1 / 3)
@inline inv(t::TaylorScalar) = one(t) / t

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = flatten(t)
            v1 = $($(QuoteNode(func)) == :expm1 ? :(exp(v[1])) : :($$func(v[1])))
        end
        for i in 2:(N + 1)
            ex = quote
                $ex
                $(Symbol('v', i)) = +($([:($(binomial(i - 2, j - 1)) * $(Symbol('v', j)) *
                                           v[$(i + 1 - j)])
                                         for j in 1:(i - 1)]...))
            end
            if $(QuoteNode(func)) == :exp2
                ex = :($ex; $(Symbol('v', i)) *= $(log(2)))
            elseif $(QuoteNode(func)) == :exp10
                ex = :($ex; $(Symbol('v', i)) *= $(log(10)))
            end
        end
        if $(QuoteNode(func)) == :expm1
            ex = :($ex; v1 = expm1(v[1]))
        end
        ex = :($ex; TaylorScalar(tuple($([Symbol('v', i) for i in 1:(N + 1)]...))))
        return :(@inbounds $ex)
    end
end

for func in (:sin, :cos)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = flatten(t)
            s1 = sin(v[1])
            c1 = cos(v[1])
        end
        for i in 2:(N + 1)
            ex = :($ex;
            $(Symbol('s', i)) = +($([:($(binomial(i - 2, j - 1)) *
                                       $(Symbol('c', j)) *
                                       v[$(i + 1 - j)]) for j in 1:(i - 1)]...)))
            ex = :($ex;
            $(Symbol('c', i)) = +($([:($(-binomial(i - 2, j - 1)) *
                                       $(Symbol('s', j)) *
                                       v[$(i + 1 - j)]) for j in 1:(i - 1)]...)))
        end
        if $(QuoteNode(func)) == :sin
            ex = :($ex; TaylorScalar(tuple($([Symbol('s', i) for i in 1:(N + 1)]...))))
        else
            ex = :($ex; TaylorScalar(tuple($([Symbol('c', i) for i in 1:(N + 1)]...))))
        end
        return quote
            @inbounds $ex
        end
    end
end

@inline sinpi(t::TaylorScalar) = sin(π * t)
@inline cospi(t::TaylorScalar) = cos(π * t)

# Binary

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

@generated function *(a::TaylorScalar{T, N}, b::TaylorScalar{T, N}) where {T, N}
    return quote
        va, vb = flatten(a), flatten(b)
        r = tuple($([:(+($([:($(binomial(i - 1, j - 1)) * va[$j] *
                              vb[$(i + 1 - j)]) for j in 1:i]...)))
                     for i in 1:(N + 1)]...))
        @inbounds TaylorScalar(r[1], r[2:end])
    end
end

@generated function /(a::TaylorScalar{T, N}, b::TaylorScalar{T, N}) where {T, N}
    ex = quote
        va, vb = flatten(a), flatten(b)
        v1 = va[1] / vb[1]
    end
    for i in 2:(N + 1)
        ex = quote
            $ex
            $(Symbol('v', i)) = (va[$i] -
                                 +($([:($(binomial(i - 1, j - 1)) * $(Symbol('v', j)) *
                                        vb[$(i + 1 - j)])
                                      for j in 1:(i - 1)]...))) / vb[1]
        end
    end
    ex = quote
        $ex
        v = tuple($([Symbol('v', i) for i in 1:(N + 1)]...))
        TaylorScalar(v)
    end
    return :(@inbounds $ex)
end

for R in (Integer, Real)
    @eval @generated function ^(t::TaylorScalar{T, N}, n::S) where {S <: $R, T, N}
        ex = quote
            v = flatten(t)
            w11 = 1
            u1 = ^(v[1], n)
        end
        for k in 1:(N + 1)
            ex = quote
                $ex
                $(Symbol('p', k)) = ^(v[1], n - $(k - 1))
            end
        end
        for i in 2:(N + 1)
            subex = quote
                $(Symbol('w', i, 1)) = 0
            end
            for k in 2:i
                subex = quote
                    $subex
                    $(Symbol('w', i, k)) = +($([:((n * $(binomial(i - 2, j - 1)) -
                                                   $(binomial(i - 2, j - 2))) *
                                                  $(Symbol('w', j, k - 1)) *
                                                  v[$(i + 1 - j)])
                                                for j in (k - 1):(i - 1)]...))
                end
            end
            ex = quote
                $ex
                $subex
                $(Symbol('u', i)) = +($([:($(Symbol('w', i, k)) * $(Symbol('p', k)))
                                         for k in 2:i]...))
            end
        end
        ex = quote
            $ex
            v = tuple($([Symbol('u', i) for i in 1:(N + 1)]...))
            TaylorScalar(v)
        end
        return :(@inbounds $ex)
    end
    @eval function ^(a::S, t::TaylorScalar{T, N}) where {S <: $R, T, N}
        exp(t * log(a))
    end
end

^(t::TaylorScalar, s::TaylorScalar) = exp(s * log(t))

@generated function raise(f::T, df::TaylorScalar{T, M},
        t::TaylorScalar{T, N}) where {T, M, N} # M + 1 == N
    return quote
        $(Expr(:meta, :inline))
        vdf, vt = flatten(df), flatten(t)
        partials = tuple($([:(+($([:($(binomial(i - 1, j - 1)) * vdf[$j] *
                                     vt[$(i + 2 - j)]) for j in 1:i]...)))
                            for i in 1:(M + 1)]...))
        @inbounds TaylorScalar(f, partials)
    end
end

raise(::T, df::S, t::TaylorScalar{T, N}) where {S <: Number, T, N} = df * t

@generated function raiseinv(f::T, df::TaylorScalar{T, M},
        t::TaylorScalar{T, N}) where {T, M, N} # M + 1 == N
    ex = quote
        vdf, vt = flatten(df), flatten(t)
        v1 = vt[2] / vdf[1]
    end
    for i in 2:(M + 1)
        ex = quote
            $ex
            $(Symbol('v', i)) = (vt[$(i + 1)] -
                                 +($([:($(binomial(i - 1, j - 1)) * $(Symbol('v', j)) *
                                        vdf[$(i + 1 - j)])
                                      for j in 1:(i - 1)]...))) / vdf[1]
        end
    end
    ex = quote
        $ex
        v = tuple($([Symbol('v', i) for i in 1:(M + 1)]...))
        TaylorScalar(f, v)
    end
    return :(@inbounds $ex)
end
