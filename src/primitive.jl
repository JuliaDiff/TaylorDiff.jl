import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch
import Base: sinc, cosc
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min

# Unary

## Delegated

@inline sqrt(t::TaylorScalar) = t^0.5
@inline cbrt(t::TaylorScalar) = ^(t, 1 / 3)
@inline inv(t::TaylorScalar) = one(t) / t

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = value(t)
            v1 = $($(QuoteNode(func)) == :expm1 ? :(exp(v[1])) : :($$func(v[1])))
        end
        for i in 2:N
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
        ex = :($ex; TaylorScalar{T, N}(tuple($([Symbol('v', i) for i in 1:N]...))))
        return :(@inbounds $ex)
    end
end

for func in (:sin, :cos)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = value(t)
            s1 = sin(v[1])
            c1 = cos(v[1])
        end
        for i in 2:N
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
            ex = :($ex; TaylorScalar($([Symbol('s', i) for i in 1:N]...)))
        else
            ex = :($ex; TaylorScalar($([Symbol('c', i) for i in 1:N]...)))
        end
        return quote
            @inbounds $ex
        end
    end
end

# Binary

for op in [:>, :<, :(==), :(>=), :(<=)]
    @eval @inline $op(a::Number, b::TaylorScalar) = $op(a, value(b)[1])
    @eval @inline $op(a::TaylorScalar, b::Number) = $op(value(a)[1], b)
    @eval @inline $op(a::TaylorScalar, b::TaylorScalar) = $op(value(a)[1], value(b)[1])
end

@inline +(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(map(+, value(a), value(b)))
@inline -(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(map(-, value(a), value(b)))

@generated function *(a::TaylorScalar{T, N}, b::TaylorScalar{T, N}) where {T, N}
    return quote
        va, vb = value(a), value(b)
        @inbounds TaylorScalar($([:(+($([:($(binomial(i - 1, j - 1)) * va[$j] *
                                           vb[$(i + 1 - j)]) for j in 1:i]...)))
                                  for i in 1:N]...))
    end
end

@generated function /(a::TaylorScalar{T, N}, b::TaylorScalar{T, N}) where {T, N}
    ex = quote
        va, vb = value(a), value(b)
        v1 = va[1] / vb[1]
    end
    for i in 2:N
        ex = quote
            $ex
            $(Symbol('v', i)) = (va[$i] -
                                 +($([:($(binomial(i - 1, j - 1)) * $(Symbol('v', j)) *
                                        vb[$(i + 1 - j)])
                                      for j in 1:(i - 1)]...))) / vb[1]
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end
@inline *(a::TaylorScalar{T1, N}, b::TaylorScalar{T2, N}) where {T1,T2,N} = *(promote(a,b)...)
@inline /(a::TaylorScalar{T1, N}, b::TaylorScalar{T2, N}) where {T1,T2,N} = *(promote(a,b)...)

@generated function ^(t::TaylorScalar{T, N}, n::S) where {S <: Number, T, N}
    ex = quote
        v = value(t)
        v1 = ^(v[1], n)
    end
    for i in 2:N
        ex = quote
            $ex
            $(Symbol('v', i)) = +($([:((n * $(binomial(i - 2, j - 1)) -
                                        $(binomial(i - 2, j - 2))) * $(Symbol('v', j)) *
                                       v[$(i + 1 - j)])
                                     for j in 1:(i - 1)]...)) / v[1]
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

@generated function ^(t::TaylorScalar{T, N}, n::S) where {S <: Integer, T, N}
    # TODO: optimize for small powers
    ex = quote
        v = value(t)
        v1 = ^(v[1], n)
    end
    for i in 2:N
        ex = quote
            $ex
            $(Symbol('v', i)) = +($([:((n * $(binomial(i - 2, j - 1)) -
                                        $(binomial(i - 2, j - 2))) * $(Symbol('v', j)) *
                                       v[$(i + 1 - j)])
                                     for j in 1:(i - 1)]...)) / v[1]
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

@generated function raise(f::T, df::TaylorScalar{T, M},
                          t::TaylorScalar{T, N}) where {T, M, N} # M + 1 == N
    return quote
        $(Expr(:meta, :inline))
        vdf, vt = value(df), value(t)
        @inbounds TaylorScalar(f,
                               $([:(+($([:($(binomial(i - 1, j - 1)) * vdf[$j] *
                                           vt[$(i + 2 - j)]) for j in 1:i]...)))
                                  for i in 1:M]...))
    end
end

raise(::T, df::S, t::TaylorScalar{T, N}) where {S <: Number, T, N} = df * t

@generated function raiseinv(f::T, df::TaylorScalar{T, M},
                             t::TaylorScalar{T, N}) where {T, M, N} # M + 1 == N
    ex = quote
        vdf, vt = value(df), value(t)
        v1 = vt[2] / vdf[1]
    end
    for i in 2:M
        ex = quote
            $ex
            $(Symbol('v', i)) = (vt[$(i + 1)] -
                                 +($([:($(binomial(i - 1, j - 1)) * $(Symbol('v', j)) *
                                        vdf[$(i + 1 - j)])
                                      for j in 1:(i - 1)]...))) / vdf[1]
        end
    end
    ex = :($ex; TaylorScalar(f, $([Symbol('v', i) for i in 1:M]...)))
    return :(@inbounds $ex)
end
