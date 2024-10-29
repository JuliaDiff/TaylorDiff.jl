export derivative, derivative!, derivatives

# Added to help Zygote infer types
@inline make_seed(x::T, l::T, ::Val{P}) where {T <: Real, P} = TaylorScalar{P}(x, l)
@inline make_seed(x::A, l::A, p) where {A <: AbstractArray} = broadcast(make_seed, x, l, p)

"""
    derivative(f, x, ::Val{P})
    derivative(f, x, l, ::Val{P})
    derivative(f!, y, x, l, ::Val{P})

Computes `P`-th directional derivative of `f` w.r.t. vector `x` in direction `l`. If `x` is a Number, the direction `l` can be omitted.
"""
function derivative end

@inline derivative(f, x::Number, p) = extract_derivative(derivatives(f, x, one(x), p), p)
@inline derivative(f, x, l, p) = extract_derivative(derivatives(f, x, l, p), p)
@inline derivative(f!, y, x, l, p) = extract_derivative(derivatives(f!, y, x, l, p), p)

"""
    derivative!(result, f, x, l, ::Val{P})
    derivative!(result, f!, y, x, l, ::Val{P})

In-place derivative calculation APIs. `result` is expected to be pre-allocated and have the same shape as `y`.
"""
function derivative! end

@inline derivative!(result, f, x, l, p) = extract_derivative!(
    result, derivatives(f, x, l, p), p)
@inline derivative!(result, f!, y, x, l, p) = extract_derivative!(
    result, derivatives(f!, y, x, l, p), p)

"""
    derivatives(f, x, l, ::Val{P})
    derivatives(f!, y, x, l, ::Val{P})

Computes all derivatives of `f` at `x` up to order `P`.
"""
function derivatives end

@inline derivatives(f, x, l, p) = f(make_seed(x, l, p))
@inline function derivatives(f!, y, x, l, p::Val{P}) where {P}
    buffer = similar(y, TaylorScalar{eltype(y), P})
    f!(buffer, make_seed(x, l, p))
    map!(value, y, buffer)
    return buffer
end
