
using SliceMap
export derivative

"""
    derivative(f, x::T, order::Int64)
    derivative(f, x::T, ::Val{N})

Computes `order`-th derivative of `f` w.r.t. scalar `x`.

    derivative(f, x::AbstractVector{T}, l::AbstractVector{T}, order::Int64)
    derivative(f, x::AbstractVector{T}, l::AbstractVector{T}, ::Val{N})

Computes `order`-th directional derivative of `f` w.r.t. vector `x` in direction `l`.

    derivative(f, x::AbstractMatrix{T}, order::Int64)
    derivative(f, x::AbstractMatrix{T}, ::Val{N})
    derivative(f, x::AbstractMatrix{T}, l::AbstractVector{T}, order::Int64)
    derivative(f, x::AbstractMatrix{T}, l::AbstractVector{T}, ::Val{N})

Shorthand notations for multiple calculations.
For a M-by-N matrix, calculate the directional derivative for each column.
For a 1-by-N matrix (row vector), calculate the derivative for each scalar.
"""
function derivative end

# Convenience wrappers for converting orders to value types
# and forward work to core APIs

@inline function derivative(f, x, order::Int64)
    derivative(f, x, Val{order + 1}())
end

@inline function derivative(f, x, l, order::Int64)
    derivative(f, x, l, Val{order + 1}())
end

# Core APIs

# Added to help Zygote infer types
make_taylor(t0::T, t1::S, ::Val{N}) where {T, S, N} = TaylorScalar{T, N}(t0, T(t1))

@inline function derivative(f, x::T, ::Val{N}) where {T <: TN, N}
    t = TaylorScalar{T, N}(x, one(x))
    return extract_derivative(f(t), N)
end

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{S},
    vN::Val{N}) where {T <: TN, S <: TN, N}
    t = map((t0, t1) -> make_taylor(t0, t1, vN), x, l)
    # equivalent to map(TaylorScalar{T, N}, x, l)
    return extract_derivative(f(t), N)
end

# shorthand notations for matrices

@inline function derivative(f, x::AbstractMatrix{T}, vN::Val{N}) where {T <: TN, N}
    size(x)[1] != 1 && @warn "x is not a row vector."
    mapcols(u -> derivative(f, u[1], vN), x)
end

@inline function derivative(f, x::AbstractMatrix{T}, l::AbstractVector{S},
    vN::Val{N}) where {T <: TN, S <: TN, N}
    mapcols(u -> derivative(f, u, l, vN), x)
end
