
import Base: getindex, setindex!, size, *

struct TaylorVector{T<:Number,N} <: AbstractArray{T, 1}
    value::NTuple{N,Vector{T}}
end

TaylorVector(x::Vararg{T,N}) where {T<:Vector,N} = TaylorVector(x)

value(tv::TaylorVector) = tv.value

Base.size(tv::TaylorVector) = Base.size(value(tv)[1])
Base.IndexStyle(::Type{<:TaylorVector}) = IndexLinear()

function getindex(tv::TaylorVector{T,N}, i::Int) where {T,N}
    v = value(tv)
    return Taylor([getindex(vec, i) for vec in v]...)
end

function setindex!(tv::TaylorVector{T,N}, t::Taylor{T,N}, i::Int) where {T,N}
    vval = value(tv)
    val = value(t)
    for j in 1:N
        setindex!(vval[j], val[j], i)
    end
end

function *(A::Matrix{T}, x::TaylorVector{T,N}) where {T,N}
    v = value(x)
    return TaylorVector([A * vec for vec in v]...)
end

function *(a::T, x::TaylorVector{T,N}) where {T<:Number,N}
    v = value(x)
    return TaylorVector([a * vec for vec in v]...)
end

function +(a::Vector{T}, x::TaylorVector{T,N}) where {T<:Number,N}
    v = value(x)
    return TaylorVector([a + vec for vec in v]...)
end
