
export derivative

@inline derivative(f::Function, x::T, order::Int64) where {T<:Number} = derivative(f, x, Val{order + 1}())

@inline function derivative(f::Function, x::T, ::Val{N}) where {T<:Number, N}
    t = TaylorScalar{T,N}(x)
    return getindex(value(f(t)), N)
end

@inline function derivative(f::Function, x::Vector{T}, index::Int64, order::Int64) where {T<:Number}
    pertub = [ind == index ? one(T) : zero(T) for ind in eachindex(x)]
    t = TaylorVector(x, pertub, [zeros(T, length(x)) for i in 1:(order - 1)]...)
    result = f(t)
    return getindex(value(result), order + 1)
end
