
import Base: getindex, setindex!, size, *, broadcast

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

function *(a::S, x::TaylorVector{T,N}) where {S<:Number,T<:Number,N}
    v = value(x)
    return TaylorVector([a * vec for vec in v]...)
end

*(x::S, a::TaylorVector{T,N}) where {S<:Number,T<:Number,N} = *(a, x)

function +(a::Vector{T}, x::TaylorVector{T,N}) where {T<:Number,N}
    v = value(x)
    return TaylorVector([a + vec for vec in v]...)
end

Base.BroadcastStyle(::Type{<:TaylorVector}) = Broadcast.ArrayStyle{TaylorVector}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TaylorVector}}, ::Type{ElType}) where ElType
    N = container_info(bc)
    TaylorVector([similar(Array{ElType}, axes(bc)) for i = 1:N]...)
end

container_info(bc::Base.Broadcast.Broadcasted) = container_info(bc.args)
container_info(args::Tuple) = container_info(container_info(args[1]), Base.tail(args))
container_info(x) = x
container_info(::Tuple{}) = nothing
container_info(::TaylorVector{T,N}, rest) where {T,N} = N
container_info(::Any, rest) = container_info(rest)

# function broadcast(f::typeof(+), x::TaylorVector{T,N}, a::Vararg) where {T<:Number,N}
#     v = value(x)
#     return TaylorVector([broadcast(f, vec, a) for vec in v]...)
# end
