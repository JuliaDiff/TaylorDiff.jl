
import Base: getindex, setindex!, size, *
import Base: BroadcastStyle, broadcasted, similar

export TaylorVector

struct TaylorVector{T<:Number,N} <: AbstractVector{T}
    value::NTuple{N,Vector{T}}
end

TaylorVector(xs::Vararg{T,N}) where {T<:Vector,N} = TaylorVector(xs)
@adjoint TaylorVector(v) = TaylorVector(v), t̄ -> (t̄.value,)
value(t::TaylorVector) = t.value
@adjoint value(t::TaylorVector) = value(t), v̄ -> (TaylorVector(v̄),)

Base.size(t::TaylorVector) = Base.size(value(t)[1])
Base.IndexStyle(::Type{<:TaylorVector}) = IndexLinear()

function getindex(t::TaylorVector{T,N}, i::Int) where {T,N}
    v = value(t)
    return TaylorScalar([getindex(vec, i) for vec in v]...)
end

elementvector(x::T, i, n) where {T} = [zeros(T,i-1); [x]; zeros(T, n-i)]

@adjoint getindex(ts::TaylorVector{T,N}, i::Int) where {N, T<:Number} = begin
    getindex(ts, i), t̄ -> (
        let v̄ = value(t̄)
            TaylorVector([elementvector(x, i, length(ts)) for x in v̄]...)
        end
    , nothing)
end

function setindex!(ts::TaylorVector{T,N}, t::TaylorScalar{T,N}, i::Int) where {T,N}
    vs, v = value(ts), value(t)
    for j in 1:N
        setindex!(vs[j], v[j], i)
    end
end

function *(A::AbstractMatrix, t::TaylorVector{T,N}) where {T,N}
    v = value(t)
    return TaylorVector([A * vec for vec in v]...)
end

# function *(a::S, t::TaylorVector{T,N}) where {S<:Number,T<:Number,N}
#     v = value(t)
#     return TaylorVector([a * vec for vec in v]...)
# end

# *(t::TaylorVector{T,N}, a::S) where {S<:Number,T<:Number,N} = *(a, t)

BroadcastStyle(::Type{<:TaylorVector}) = Broadcast.ArrayStyle{TaylorVector}()

function similar(t::TaylorVector{T,N}) where {T, N}
    v = value(t)
    TaylorVector([similar(vec) for vec in v]...)
end

function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TaylorVector}}, ::Type{ElType}) where ElType
    t = container_info(bc)
    similar(t)
end

container_info(bc::Base.Broadcast.Broadcasted) = container_info(bc.args)
container_info(args::Tuple) = begin
    container_info(container_info(args[1]), Base.tail(args))
end

container_info(x) = x
container_info(::Tuple{}) = nothing
container_info(t::TaylorVector{T,N}, rest...) where {T,N} = t
container_info(::Any, rest...) = container_info(rest...)

broadcasted(::Broadcast.ArrayStyle{TaylorVector}, ::typeof(sin), t::TaylorVector) = "Hello"
