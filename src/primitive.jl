
import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch, sinc, cosc
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min

Taylor{T,N} = Union{TaylorScalar{T,N}, TaylorVector{T,N}}

# Unary

+(t::Taylor) = t
-(t::Taylor) = -1 * t
sqrt(t::Taylor) = ^(t, .5)
cbrt(t::Taylor) = ^(t, 1/3)
inv(t::Taylor) = ^(t, -1)

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::TaylorScalar{T,N}) where {T, N}
        ex = quote
            v = value(t)
            v1 = exp(v[1])
        end
        for i = 2:N
            ex = quote
                $ex
                $(Symbol('v', i)) = +($([
                    :($(binomial(i - 2, j - 1)) * $(Symbol('v', j)) * v[$i + 1 - $j])
                for j = 1:i-1]...))
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
    @eval @generated function $func(t::TaylorScalar{T,N}) where {T, N}
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
        for i = 3:N
            ex = quote
                $ex
                $(Symbol('v', i)) = (v[$i] - +($([
                    :($(binomial(i - 2, j - 2)) * $(Symbol('v', j)) * v[$i + 1 - $j])
                for j = 2:i-1]...))) / den
            end
        end
        ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
        return :(@inbounds $ex)
    end
end

for func in (:sin, :cos)
    @eval @generated function $func(t::TaylorScalar{T,N}) where {T, N}
        ex = quote
            v = value(t)
            s1 = sin(v[1])
            c1 = cos(v[1])
        end
        for i = 2:N
            ex = quote
                $ex
                $(Symbol('s', i)) = +($([
                    :($(binomial(i - 2, j - 1)) * $(Symbol('c', j)) * v[$i + 1 - $j])
                for j = 1:i-1]...))
                $(Symbol('c', i)) = +($([
                    :($(-binomial(i - 2, j - 1)) * $(Symbol('s', j)) * v[$i + 1 - $j])
                for j = 1:i-1]...))
            end
        end
        if $(QuoteNode(func)) == :sin
            ex = :($ex; TaylorScalar($([Symbol('s', i) for i in 1:N]...)))
        else
            ex = :($ex; TaylorScalar($([Symbol('c', i) for i in 1:N]...)))
        end
        return :(@inbounds $ex)
    end
end

# Binary

for op in [:+, :*, :-, :/]
    @eval $op(x::Number, y::TaylorScalar{T,N}) where {T,N} = $op(promote(x, y)...)
    @eval $op(x::TaylorScalar, y::Number) = $op(promote(x, y)...)
end

for op in [:(==), :(<), :(<=)]
    @eval $op(a::Number, b::Taylor) = $op(a, value(b)[1])
    @eval $op(a::Taylor, b::Number) = $op(value(a)[1], b)
end

@generated function +(a::TaylorScalar{T,N}, b::TaylorScalar{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds TaylorScalar(
            $([:(va[$i] + vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function -(a::TaylorScalar{T,N}, b::TaylorScalar{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds TaylorScalar(
            $([:(va[$i] - vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function *(a::TaylorScalar{T,N}, b::TaylorScalar{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds TaylorScalar($([:(
            +($([:(
                $(binomial(i - 1, j - 1)) * va[$j] * vb[$(i + 1 - j)]
            ) for j = 1:i]...))
        ) for i = 1:N]...))
    end
end

@generated function /(a::TaylorScalar{T,N}, b::TaylorScalar{T,N}) where {T,N}
    ex = quote
        va, vb = value(a), value(b)
        v1 = va[1] / vb[1]
    end
    for i = 2:N
        ex = quote
            $ex
            $(Symbol('v', i)) = (va[$i] - +($([
                :($(binomial(i - 1, j)) * $(Symbol('v', j)) * vb[$i + 1 - $j])
            for j = 1:i-1]...))) / vb[1]
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

for op in [:>, :<, :(==), :(>=), :(<=)]
    @eval $op(a::TaylorScalar, b::TaylorScalar) = $op(value(a)[1], value(b)[1])
end

@generated function ^(t::TaylorScalar{T,N}, n::Number) where {T, N}
    ex = quote
        v = value(t)
        v1 = ^(v[1], n)
    end
    for i = 2:N
        ex = quote
            $ex
            $(Symbol('v', i)) = +($([
                :((n * $(binomial(i - 2, j - 1)) - $(binomial(i - 2, j - 2))) * $(Symbol('v', j)) * v[$i + 1 - $j])
            for j = 1:i-1]...))
        end
    end
    ex = :($ex; TaylorScalar($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end
