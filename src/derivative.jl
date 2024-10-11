
export derivative, derivative!, derivatives, make_seed

"""
    derivative(f, x, l, ::Val{P})
    derivative(f!, y, x, l, ::Val{P})

Computes `P`-th directional derivative of `f` w.r.t. vector `x` in direction `l`.
"""
function derivative end

"""
    derivative!(result, f, x, l, ::Val{P})
    derivative!(result, f!, y, x, l, ::Val{P})

In-place derivative calculation APIs. `result` is expected to be pre-allocated and have the same shape as `y`.
"""
function derivative! end

"""
    derivatives(f, x, l, ::Val{P})
    derivatives(f!, y, x, l, ::Val{P})

Computes all derivatives of `f` at `x` up to order `P`.
"""
function derivatives end

# Convenience wrapper for adding unit seed to the input

@inline derivative(f, x, p::Int64) = derivative(f, x, broadcast(one, x), p)

# Convenience wrappers for converting ps to value types
# and forward work to core APIs

@inline derivative(f, x, l, p::Int64) = derivative(f, x, l, Val{p}())
@inline derivative(f!, y, x, l, p::Int64) = derivative(f!, y, x, l, Val{p}())
@inline derivative!(result, f, x, l, p::Int64) = derivative!(
    result, f, x, l, Val{p}())
@inline derivative!(result, f!, y, x, l, p::Int64) = derivative!(
    result, f!, y, x, l, Val{p}())

# Core APIs

# Added to help Zygote infer types
@inline make_seed(x::T, l::T, ::Val{P}) where {T <: Real, P} = TaylorScalar{P}(x, l)
@inline make_seed(x::A, l::A, ::Val{P}) where {A <: AbstractArray, P} = broadcast(
    make_seed, x, l, Val{P}())

# `derivative` API: computes the `P - 1`-th derivative of `f` at `x`
@inline derivative(f, x, l, p::Val{P}) where {P} = extract_derivative(
    derivatives(f, x, l, p), p)
@inline derivative(f!, y, x, l, p::Val{P}) where {P} = extract_derivative(
    derivatives(f!, y, x, l, p), p)
@inline derivative!(result, f, x, l, p::Val{P}) where {P} = extract_derivative!(
    result, derivatives(f, x, l, p), p)
@inline derivative!(result, f!, y, x, l, p::Val{P}) where {P} = extract_derivative!(
    result, derivatives(f!, y, x, l, p), p)

# `derivatives` API: computes all derivatives of `f` at `x` up to p `P - 1`

# Out-of-place function
@inline derivatives(f, x, l, p::Val{P}) where {P} = f(make_seed(x, l, p))

# In-place function
@inline function derivatives(f!, y, x, l, p::Val{P}) where {P}
    buffer = similar(y, TaylorScalar{eltype(y), P})
    f!(buffer, make_seed(x, l, p))
    map!(value, y, buffer)
    return buffer
end
