
using SliceMap
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


@inline function derivative(f, x::M, order::Int64) where {M <: AbstractMatrix{<:Number}}
    mapcols(u -> derivative(f, u[1], order), x)
end

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{S},
    order::Int64) where {T <: Number, S <: Number}
    derivative(f, x, l, Val{order + 1}())
end

@inline function derivative(f, x::M, l::L,
    order::Int64) where {M <: AbstractMatrix{<:Number}, L <: AbstractVector{<:Number}}
    mapcols(u -> derivative(f, u, l, order), x)
end

@inline function derivative(f, x::T, ::Val{N}) where {T <: Number, N}
    t = TaylorScalar{T, N}(x, one(x))
    return extract_derivative(f(t), N)
end

@inline function derivative(f, x::M, ::Val{N}) where {M <: AbstractMatrix{<:Number}, N}
    mapcols(u -> derivative(f, u[1], N), x)
end

# Need to rewrite like this to help Zygote infer types
make_taylor(t0::T, t1::S, ::Val{N}) where {T, S, N} = TaylorScalar{T, N}(t0, T(t1))

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{S},
    vN::Val{N}) where {T <: Number, S <: Number, N}
    t = map((t0, t1) -> make_taylor(t0, t1, vN), x, l) # i.e. map(TaylorScalar{T, N}, x, l)
    return extract_derivative(f(t), N)
end

@inline function derivative(f, x::M, l::V,
    vN::Val{N}) where {M <: AbstractMatrix{<:Number}, V <: AbstractVector{<:Number}, N}
    mapcols(u -> derivative(f, u, l, vN), x)
end
