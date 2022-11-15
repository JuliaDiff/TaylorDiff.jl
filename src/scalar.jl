
struct Taylor{T<:Number,N}
    value::NTuple{N,T}
end

Taylor(x::Vararg{T,N}) where {T<:Number,N} = Taylor(x)

zero(::Taylor{T,N}) where {T, N} = Taylor(zeros(T, N)...)
one(::Taylor{T,N}) where {T, N} = Taylor(one(T), zeros(T, N-1)...)

zero(::Type{Taylor{T,N}}) where {T, N} = Taylor(zeros(T, N)...)
one(::Type{Taylor{T,N}}) where {T, N} = Taylor(one(T), zeros(T, N-1)...)

value(a::Taylor) = a.value

for op in [:+, :*, :-, :/, :(==), :(<), :(<=)]
    @eval $op(x::Number, y::Taylor) = $op(promote(x, y)...)
    @eval $op(x::Taylor, y::Number) = $op(promote(x, y)...)
end

Base.adjoint(a::Taylor) = a
Base.promote_rule(::Type{Taylor{T,N}}, ::Type{S}) where {T<:Number,S<:Number,N} = Taylor{promote_type(T,S),N}
Base.convert(::Type{Taylor{T,N}}, x::T) where {T<:Number,N} = Taylor(x, zeros(T, N - 1)...)
Base.convert(::Type{Taylor{T,N}}, x::S) where {T<:Number,S<:Number,N} = Taylor(convert(T, x), zeros(T, N - 1)...)
Base.convert(::Type{Taylor{T,N}}, x::Taylor{S,N}) where {T<:Number,S<:Number,N} = Taylor([convert(T, v) for v in value(x)]...)

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

Zygote.@adjoint value(a::Taylor) = value(a), v̄ -> (Taylor(v̄),)
Zygote.@adjoint Taylor(a) = Taylor(a), ā -> (ā.value,)
Zygote.@adjoint getindex(t::NTuple{N,T}, i::Int) where {N, T<:Number} = getindex(t, i), v̄ -> (tuple(zeros(T,i-1)..., v̄, zeros(T, N-i)...), nothing)

@generated function exp(t::Taylor{T,N}) where {T, N}
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
    end
    ex = :($ex; Taylor($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

# sin(m::Taylorulti{V,N}) where {V, N} = begin
#     sv = @TaylorVector zeros(V,N)
#     cv = @TaylorVector zeros(V,N)
#     sv[1] = sin(m.value[1])
#     cv[1] = cos(m.value[1])
#     for i = 2:N
#         for j = 1:i-1
#             sv[i] += binomial(i - 2, j - 1) * cv[j] * m.value[i - j]
#             cv[i] -= binomial(i - 2, j - 1) * sv[j] * m.value[i - j]
#         end
#     end
#     Taylorulti(sv)
# end
