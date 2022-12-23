
export derivative

"""
    derivative(f, x::T, order::Int64)
    derivative(f, x::T, ::Val{N})

Computes `order`-th derivative of `f` w.r.t. `x`.

    derivative(f, x::Vector{T}, l::Vector{T}, order::Int64)
    derivative(f, x::Vector{T}, l::Vector{T}, ::Val{N})

Computes `order`-th directional derivative of `f` w.r.t. `x` in direction `l`.
"""
function derivative end

@inline function derivative(f, x::T, order::Int64) where {T <: Number}
    derivative(f, x, Val{order + 1}())
end

@inline function derivative(f, x::Vector{T}, l::Vector{T},
                            order::Int64) where {T <: Number}
    derivative(f, x, l, Val{order + 1}())
end

@inline function derivative(f, x::T, ::Val{N}) where {T <: Number, N}
    t = TaylorScalar{T, N}(x, one(x))
    return extract_derivative(f(t), N)
end

@inline function derivative(f, x::Vector{T}, l::Vector{T},
                            ::Val{N}) where {T <: Number, N}
    t = map(TaylorScalar{T, N}, x, l)
    return extract_derivative(f(t), N)
end
