
struct M{T<:Number,N}
    value::NTuple{N,T}
end

M(x::Vararg{T,N}) where {T<:Number,N} = M(x)

zero(::M{T,N}) where {T, N} = M(zeros(T, N)...)
one(::M{T,N}) where {T, N} = M(one(T), zeros(T, N-1)...)

zero(::Type{M{T,N}}) where {T, N} = M(zeros(T, N)...)
one(::Type{M{T,N}}) where {T, N} = M(one(T), zeros(T, N-1)...)

value(a::M) = a.value

for op in [:+, :*, :-, :/, :(==), :(<), :(<=)]
    @eval $op(x::Number, y::M) = $op(promote(x, y)...)
    @eval $op(x::M, y::Number) = $op(promote(x, y)...)
end

Base.adjoint(a::M) = a
Base.promote_rule(::Type{M{T,N}}, ::Type{S}) where {T<:Number,S<:Number,N} = M{promote_type(T,S),N}
Base.convert(::Type{M{T,N}}, x::T) where {T<:Number,N} = M(x, zeros(T, N - 1)...)
Base.convert(::Type{M{T,N}}, x::S) where {T<:Number,S<:Number,N} = M(convert(T, x), zeros(T, N - 1)...)
Base.convert(::Type{M{T,N}}, x::M{S,N}) where {T<:Number,S<:Number,N} = M([convert(T, v) for v in value(x)]...)

@generated function +(a::M{T,N}, b::M{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds M(
            $([:(va[$i] + vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function -(a::M{T,N}, b::M{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds M(
            $([:(va[$i] - vb[$i]) for i = 1:N]...)
        )
    end
end

@generated function *(a::M{T,N}, b::M{T,N}) where {T,N}
    return quote
        va, vb = value(a), value(b)
        @inbounds M($([:(
            +($([:(
                $(binomial(i - 1, j - 1)) * va[$j] * vb[$(i + 1 - j)]
            ) for j = 1:i]...))
        ) for i = 1:N]...))
    end
end

@generated function /(a::M{T,N}, b::M{T,N}) where {T,N}
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
    ex = :($ex; M($([Symbol('v', i) for i in 1:N]...)))
    return :(@inbounds $ex)
end

>(a::M, b::M) = >(value(a)[1], value(b)[1])
<(a::M, b::M) = <(value(a)[1], value(b)[1])
>=(a::M, b::M) = >=(value(a)[1], value(b)[1])
<=(a::M, b::M) = <=(value(a)[1], value(b)[1])
==(a::M, b::M) = ==(value(a), value(b))

Zygote.@adjoint value(a::M) = value(a), v̄ -> (M(v̄),)
Zygote.@adjoint M(a) = M(a), ā -> (ā.value,)
Zygote.@adjoint getindex(t::NTuple{N,T}, i::Int) where {N, T<:Number} = getindex(t, i), v̄ -> (tuple(zeros(T,i-1)..., v̄, zeros(T, N-i)...), nothing)


# exp(m::Multi{V,N}) where {V, N} = begin
#     value = @MVector zeros(V,N)
#     value[1] = exp(m.value[1])
#     for i = 2:N
#         for j = 1:i-1
#             value[i] += binomial(i - 2, j - 1) * value[j] * m.value[i - j]
#         end
#     end
#     Multi(value)
# end

# sin(m::Multi{V,N}) where {V, N} = begin
#     sv = @MVector zeros(V,N)
#     cv = @MVector zeros(V,N)
#     sv[1] = sin(m.value[1])
#     cv[1] = cos(m.value[1])
#     for i = 2:N
#         for j = 1:i-1
#             sv[i] += binomial(i - 2, j - 1) * cv[j] * m.value[i - j]
#             cv[i] -= binomial(i - 2, j - 1) * sv[j] * m.value[i - j]
#         end
#     end
#     Multi(sv)
# end
