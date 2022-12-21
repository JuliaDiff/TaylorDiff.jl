
import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch,
             sinc, cosc
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min

# Unary

@inline +(t::TaylorScalar) = t
@inline -(t::TaylorScalar) = -1 * t
@inline sqrt(t::TaylorScalar) = t^0.5
@inline cbrt(t::TaylorScalar) = ^(t, 1 / 3)
@inline inv(t::TaylorScalar) = 1 / t

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = value(t)
            v1 = exp(v[1])
        end
        for i in 2:N
            ex = quote
                $ex
                $(Symbol('v', i)) = +($([:($(binomial(i - 2, j - 1)) * $(Symbol('v', j)) *
                                           v[$i + 1 - $j])
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
        ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
        return :(@inbounds $ex)
    end
end

for func in (:log, :log1p, :log2, :log10)
    @eval @generated function $func(t::TaylorScalar{T, N}) where {T, N}
        ex = quote
            v = value(t)
            v1 = $$func(v[1])
        end
        if $(QuoteNode(func)) == :log1p
            ex = :($ex; den = v[1] + 1)
        else
            ex = :($ex; den = v[1])
        end
        if $(QuoteNode(func)) == :log2
            ex = :($ex; v2 = v[2] / den / $(log(2)))
        elseif $(QuoteNode(func)) == :log10
            ex = :($ex; v2 = v[2] / den / $(log(10)))
        else
            ex = :($ex; v2 = v[2] / den)
        end
        for i in 3:N
            ex = quote
                $ex
                $(Symbol('v', i)) = (v[$i] -
                                     +($([:($(binomial(i - 2, j - 2)) * $(Symbol('v', j)) *
                                            v[$i + 1 - $j])
                                          for j in 2:(i - 1)]...))) / den
            end
        end
        ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
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
            ex = quote
                $ex
                $(Symbol('s', i)) = +($([:($(binomial(i - 2, j - 1)) * $(Symbol('c', j)) *
                                           v[$i + 1 - $j])
                                         for j in 1:(i - 1)]...))
                $(Symbol('c', i)) = +($([:($(-binomial(i - 2, j - 1)) * $(Symbol('s', j)) *
                                           v[$i + 1 - $j])
                                         for j in 1:(i - 1)]...))
            end
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

for op in [:+, :*, :-, :/]
    @eval @inline function $op(x::S, y::TaylorScalar{T, N}) where {S <: Number, T, N}
        $op(promote(x, y)...)
    end
    @eval @inline function $op(x::TaylorScalar{T, N}, y::S) where {S <: Number, T, N}
        $op(promote(x, y)...)
    end
end

for op in [:(==), :(<), :(<=)]
    @eval @inline function $op(a::S, b::TaylorScalar{T, N}) where {S <: Number, T, N}
        $op(a, value(b)[1])
    end
    @eval @inline function $op(a::TaylorScalar{T, N}, b::S) where {S <: Number, T, N}
        $op(value(a)[1], b)
    end
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
                                        vb[$i + 1 - $j])
                                      for j in 1:(i - 1)]...))) / vb[1]
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

for op in [:>, :<, :(==), :(>=), :(<=)]
    @eval $op(a::TaylorScalar, b::TaylorScalar) = $op(value(a)[1], value(b)[1])
end

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
                                       v[$i + 1 - $j])
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

@generated function raiseinv(f::T, df::TaylorScalar{T, M},
                             t::TaylorScalar{T, N}) where {T, M, N} # M + 1 == N
    ex = quote
        vdf, vt = value(df), value(t)
        v1 = vt[2] / vdf[1]
    end
    for i in 2:M
        ex = quote
            $ex
            $(Symbol('v', i)) = (vt[$i + 1] -
                                 +($([:($(binomial(i - 1, j)) * $(Symbol('v', j)) *
                                        vdf[$i + 1 - $j])
                                      for j in 1:(i - 1)]...))) / vdf[1]
        end
    end
    ex = :($ex; TaylorScalar(f, $([Symbol('v', i) for i in 1:M]...)))
    return :(@inbounds $ex)
end
