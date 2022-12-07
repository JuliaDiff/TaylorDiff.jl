
export derivative

@inline derivative(f::Function, x::T, order::Int64) where {T<:Number} = derivative(f, x, Val{order + 1}())

@inline derivative(f::Function, x::Vector{T}, l::Vector{T}, order::Int64) where {T<:Number} = derivative(f, x, l, Val{order + 1}())

@inline function derivative(f::F, x::T, ::Val{N}) where {F, T<:Number, N}
    t = TaylorScalar{T,N}(x)
    return getindex(value(f(t)), N)
end

@inline function derivative(f::F, x::Vector{T}, l::Vector{T}, ::Val{N}) where {F, T<:Number, N}
    t = TaylorVector{T,N}(x, l)
    return getindex(value(f(t)), N)
end
