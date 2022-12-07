
using Zygote: @adjoint
import Base: getindex, setindex!, size, +, *
import Base: BroadcastStyle, broadcasted, similar

export TaylorVector

struct TaylorVector{T<:Number,N} <: AbstractVector{T}
    value::NTuple{N,Vector{T}}
end

TaylorVector(xs::Vararg{T,N}) where {T<:Vector,N} = TaylorVector(xs)
@adjoint TaylorVector(v) = TaylorVector(v), t̄ -> (t̄.value,)
value(t::TaylorVector) = t.value
@adjoint value(t::TaylorVector) = value(t), v̄ -> (TaylorVector(v̄),)

@generated function TaylorVector{T,N}(x::Vector{T}, l::Vector{T}) where {T<:Number, N}
    return quote
        $(Expr(:meta, :inline))
        TaylorVector((x, l, $(
            fill(:(zeros(T, length(x))), N - 2)...
        )))
    end
end

Base.size(t::TaylorVector) = Base.size(value(t)[1])
Base.IndexStyle(::Type{<:TaylorVector}) = IndexLinear()

@inline Base.@propagate_inbounds getindex(t::TaylorVector{T,N}, i::I) where {T,N,I<:Integer} = TaylorScalar{T,N}(map(x -> x[i], value(t)))

elementvector(x::T, i, n) where {T} = [zeros(T,i-1); [x]; zeros(T, n-i)]

@adjoint getindex(ts::TaylorVector{T,N}, i::Int) where {N, T<:Number} = begin
    getindex(ts, i), t̄ -> (
        let v̄ = value(t̄)
            TaylorVector([elementvector(x, i, length(ts)) for x in v̄]...)
        end
    , nothing)
end

@inline Base.@propagate_inbounds setindex!(ts::TaylorVector{T,N}, t::TaylorScalar{T,N}, i::Int) where {T,N} = map((vs, v) -> vs[i] = v, value(ts), value(t))

@inline *(A::AbstractMatrix{S}, t::TaylorVector{T,N}) where {S<:Number,T,N} = TaylorVector{T,N}(map(v -> A * v, value(t)))

@inline +(a::AbstractVector{S}, t::TaylorVector{T,N}) where {S<:Number,T,N} = TaylorVector{T,N}(map(v -> a + v, value(t)))

@inline +(t::TaylorVector{T,N}, a::AbstractVector{S}) where {S<:Number,T,N} = a + t

@inline *(a::S, t::TaylorVector{T,N}) where {S<:Number,T,N} = TaylorVector{T,N}(map(v -> a * v, value(t)))

@inline *(t::TaylorVector{T,N}, a::S) where {S<:Number,T,N} = a * t

BroadcastStyle(::Type{<:TaylorVector}) = Broadcast.ArrayStyle{TaylorVector}()

similar(t::TaylorVector{T,N}) where {T, N} = TaylorVector{T,N}(map(similar, value(t)))

similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TaylorVector}}, ::Type{ElType}) where ElType = similar(container_info(bc))

container_info(bc::Base.Broadcast.Broadcasted) = container_info(bc.args)
container_info(args::Tuple) = begin
    container_info(container_info(args[1]), Base.tail(args))
end

container_info(x) = x
container_info(::Tuple{}) = nothing
container_info(t::TaylorVector{T,N}, rest...) where {T,N} = t
container_info(::Any, rest...) = container_info(rest...)

# @inline broadcasted(::Broadcast.ArrayStyle{TaylorVector}, ::typeof(exp), t::TaylorVector{T,N}) where {T,N} = TaylorVector{T,N}(map(v -> exp.(v), value(t)))
