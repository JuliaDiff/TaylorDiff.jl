
export derivative, derivative!, derivatives, make_seed

"""
    derivative(f, x, l, ::Val{N})
    derivative(f!, y, x, l, ::Val{N})

Computes `order`-th directional derivative of `f` w.r.t. vector `x` in direction `l`.
"""
function derivative end

"""
    derivative!(result, f, x, l, ::Val{N})
    derivative!(result, f!, y, x, l, ::Val{N})

In-place derivative calculation APIs. `result` is expected to be pre-allocated and have the same shape as `y`.
"""
function derivative! end

"""
    derivatives(f, x, l, ::Val{N})
    derivatives(f!, y, x, l, ::Val{N})

Computes all derivatives of `f` at `x` up to order `N - 1`.
"""
function derivatives end

# Convenience wrapper for adding unit seed to the input

@inline derivative(f, x, order::Int64) = derivative(f, x, one(eltype(x)), order)

# Convenience wrappers for converting orders to value types
# and forward work to core APIs

@inline derivative(f, x, l, order::Int64) = derivative(f, x, l, Val{order + 1}())
@inline derivative(f!, y, x, l, order::Int64) = derivative(f!, y, x, l, Val{order + 1}())
@inline derivative!(result, f, x, l, order::Int64) = derivative!(
    result, f, x, l, Val{order + 1}())
@inline derivative!(result, f!, y, x, l, order::Int64) = derivative!(
    result, f!, y, x, l, Val{order + 1}())

# Core APIs

# Added to help Zygote infer types
@inline function make_seed(x::T, l::S, ::Val{N}) where {T <: TN, S <: TN, N}
    TaylorScalar{T, N}(x, convert(T, l))
end

@inline function make_seed(x::AbstractArray{T}, l, vN::Val{N}) where {T <: TN, N}
    broadcast(make_seed, x, l, vN)
end

# `derivative` API: computes the `N - 1`-th derivative of `f` at `x`
@inline derivative(f, x, l, vN::Val{N}) where {N} = extract_derivative(
    derivatives(f, x, l, vN), N)
@inline derivative(f!, y, x, l, vN::Val{N}) where {N} = extract_derivative(
    derivatives(f!, y, x, l, vN), N)
@inline derivative!(result, f, x, l, vN::Val{N}) where {N} = extract_derivative!(
    result, derivatives(f, x, l, vN), N)
@inline derivative!(result, f!, y, x, l, vN::Val{N}) where {N} = extract_derivative!(
    result, derivatives(f!, y, x, l, vN), N)

# `derivatives` API: computes all derivatives of `f` at `x` up to order `N - 1`

# Out-of-place function
@inline derivatives(f, x, l, vN::Val{N}) where {N} = f(make_seed(x, l, vN))

# In-place function
@inline function derivatives(f!, y::AbstractArray{T}, x, l, vN::Val{N}) where {T, N}
    buffer = similar(y, TaylorScalar{T, N})
    f!(buffer, make_seed(x, l, vN))
    map!(primal, y, buffer)
    return buffer
end
