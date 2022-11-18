
struct Taylor{T<:Number,N}
    value::NTuple{N,T}
end

Taylor(x::Vararg{T,N}) where {T<:Number,N} = Taylor(x)

zero(::Taylor{T,N}) where {T, N} = Taylor(zeros(T, N)...)
one(::Taylor{T,N}) where {T, N} = Taylor(one(T), zeros(T, N-1)...)

zero(::Type{Taylor{T,N}}) where {T, N} = Taylor(zeros(T, N)...)
one(::Type{Taylor{T,N}}) where {T, N} = Taylor(one(T), zeros(T, N-1)...)

value(a::Taylor) = a.value

for op in [:+, :*, :-, :/]
    @eval $op(x::Number, y::Taylor{T,N}) where {T,N} = $op(promote(x, y)...)
    @eval $op(x::Taylor, y::Number) = $op(promote(x, y)...)
end

for op in [:(==), :(<), :(<=)]
    @eval $op(x::Number, y::Taylor) = $op(x, value(y)[1])
    @eval $op(x::Taylor, y::Number) = $op(value(x)[1], y)
end

Base.adjoint(a::Taylor) = a
Base.promote_rule(::Type{Taylor{T,N}}, ::Type{S}) where {T<:Number,S<:Number,N} = Taylor{promote_type(T,S),N}
Base.convert(::Type{Taylor{T,N}}, x::T) where {T<:Number,N} = Taylor(x, zeros(T, N - 1)...)
Base.convert(::Type{Taylor{T,N}}, x::S) where {T<:Number,S<:Number,N} = Taylor(convert(T, x), zeros(T, N - 1)...)
Base.convert(::Type{Taylor{T,N}}, x::Taylor{S,N}) where {T<:Number,S<:Number,N} = Taylor([convert(T, v) for v in value(x)]...)

+(a::Taylor) = a
-(a::Taylor) = -1 * a

@generated function +(a::Taylor{T,N}, b::Taylor{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds Taylor(
            $([:(va[$i] + vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function -(a::Taylor{T,N}, b::Taylor{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds Taylor(
            $([:(va[$i] - vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function *(a::Taylor{T,N}, b::Taylor{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds Taylor($([:(
            +($([:(
                $(binomial(i - 1, j - 1)) * va[$j] * vb[$(i + 1 - j)]
            ) for j = 1:i]...))
        ) for i = 1:N]...))
    end
end

@generated function /(a::Taylor{T,N}, b::Taylor{T,N}) where {T,N}
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
    ex = :($ex; Taylor($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

for op in [:>, :<, :(==), :(>=), :(<=)]
    @eval $op(a::Taylor, b::Taylor) = $op(value(a)[1], value(b)[1])
end

@generated function ^(t::Taylor{T,N}, n::Number) where {T, N}
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
    ex = :($ex; Taylor($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

sqrt(t::Taylor) = ^(t, .5)
cbrt(t::Taylor) = ^(t, 1/3)
inv(t::Taylor) = ^(t, -1)

Zygote.@adjoint value(a::Taylor) = value(a), v̄ -> (Taylor(v̄),)
Zygote.@adjoint Taylor(a) = Taylor(a), ā -> (ā.value,)
Zygote.@adjoint getindex(t::NTuple{N,T}, i::Int) where {N, T<:Number} = getindex(t, i), v̄ -> (tuple(zeros(T,i-1)..., v̄, zeros(T, N-i)...), nothing)

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::Taylor{T,N}) where {T, N}
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
        ex = :($ex; Taylor($([Symbol('v', i) for i in 1:N]...)))
        return :(@inbounds $ex)
    end
end

for func in (:log, :log1p, :log2, :log10)
    @eval @generated function $func(t::Taylor{T,N}) where {T, N}
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
        ex = :($ex; Taylor($([Symbol('v', i) for i in 1:N]...)))
        return :(@inbounds $ex)
    end
end

for func in (:sin, :cos)
    @eval @generated function $func(t::Taylor{T,N}) where {T, N}
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
            ex = :($ex; Taylor($([Symbol('s', i) for i in 1:N]...)))
        else
            ex = :($ex; Taylor($([Symbol('c', i) for i in 1:N]...)))
        end
        return :(@inbounds $ex)
    end
end
