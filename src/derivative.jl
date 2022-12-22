
export derivative

"""
    derivative(f::Function, x::T, order::Int64)
    derivative(f::F, x::T, ::Val{N})

Computes `order`-th derivative of `f` w.r.t. `x`.

    derivative(f::Function, x::Vector{T}, l::Vector{T}, order::Int64)
    derivative(f::F, x::Vector{T}, l::Vector{T}, ::Val{N})

Computes `order`-th directional derivative of `f` w.r.t. `x` in direction `l`.
"""
function derivative end

@inline function derivative(f::Function, x::T, order::Int64) where {T <: Number}
    derivative(f, x, Val{order + 1}())
end

@inline function derivative(f::Function, x::Vector{T}, l::Vector{T},
                            order::Int64) where {T <: Number}
    derivative(f, x, l, Val{order + 1}())
end

@inline function derivative(f::F, x::T, ::Val{N}) where {F, T <: Number, N}
    t = TaylorScalar{T, N}(x)
    return extract_derivative(f(t), N)
end

@inline function derivative(f::F, x::Vector{T}, l::Vector{T},
                            ::Val{N}) where {F, T <: Number, N}
    t = map(TaylorScalar{T, N}, x, l)
    return extract_derivative(f(t), N)
end
