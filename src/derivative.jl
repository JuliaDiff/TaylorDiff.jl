
export derivative, derivative!

"""
    derivative(f, x, order::Int64)
    derivative(f, x, l, order::Int64)

Wrapper functions for converting order from a number to a type. Actual APIs are detailed below:

    derivative(f, x::T, ::Val{N})

Computes `order`-th derivative of `f` w.r.t. scalar `x`.

    derivative(f, x::AbstractVector{T}, l::AbstractVector{T}, ::Val{N})

Computes `order`-th directional derivative of `f` w.r.t. vector `x` in direction `l`.

    derivative(f, x::AbstractMatrix{T}, ::Val{N})
    derivative(f, x::AbstractMatrix{T}, l::AbstractVector{T}, ::Val{N})

Batch mode derivative / directional derivative calculations, where each column of `x` represents a scalar or a vector. `f` is expected to accept matrices as input.
- For a M-by-N matrix, calculate the directional derivative for each column.
- For a 1-by-N matrix (row vector), calculate the derivative for each scalar.
"""
function derivative end

"""
    derivative!(result, f, x, l, ::Val{N})
    derivative!(result, f!, y, x, l, ::Val{N})

In-place derivative calculation APIs. `result` is expected to be pre-allocated and have the same shape as `y`.
"""
function derivative! end

# Convenience wrappers for converting orders to value types
# and forward work to core APIs

@inline derivative(f, x, order::Int64) = derivative(f, x, one(eltype(x)), order)
@inline derivative(f, x, l, order::Int64) = derivative(f, x, l, Val{order + 1}())

# Core APIs

# Added to help Zygote infer types
@inline function make_taylor(x::T, l::S, ::Val{N}) where {T <: TN, S <: TN, N}
    TaylorScalar{T, N}(x, convert(T, l))
end

@inline function make_taylor(x::AbstractArray{T}, l, vN::Val{N}) where {T <: TN, N}
    broadcast(make_taylor, x, l, vN)
end

# Out-of-place function, out-of-place derivative
@inline function derivative(f, x, l, vN::Val{N}) where {N}
    t = make_taylor(x, l, vN)
    return extract_derivative(f(t), N)
end

# Below three advanced APIs do not have convenience wrappers

# In-place function, out-of-place derivative
@inline function derivative(f!, y::AbstractArray{T}, x, l, vN::Val{N}) where {T, N}
    s = similar(y, TaylorScalar{T, N})
    t = make_taylor(x, l, vN)
    f!(s, t)
    map!(primal, y, s)
    return extract_derivative(s, N)
end

# Out-of-place function, in-place derivative
@inline function derivative!(result, f, x, l, vN::Val{N}) where {N}
    t = make_taylor(x, l, vN)
    s = f(t)
    extract_derivative!(result, s, N)
    return result
end

# In-place function, in-place derivative
@inline function derivative!(result, f!, y::AbstractArray{T}, x, l, vN::Val{N}) where {T, N}
    s = similar(y, TaylorScalar{T, N})
    t = make_taylor(x, l, vN)
    f!(s, t)
    map!(primal, y, s)
    extract_derivative!(result, s, N)
    return result
end
