export TaylorArray

"""
    TaylorArray{T, N, A, P}

Representation of Taylor polynomials in array mode.

# Fields

- `value::A`: zeroth order coefficient
- `partials::NTuple{P, T}`: i-th element of this stores the i-th derivative
"""
struct TaylorArray{T, N, A <: AbstractArray{T, N}, P} <:
       AbstractArray{TaylorScalar{T, P}, N}
    value::A
    partials::NTuple{P, A}
    function TaylorArray(
            value::A, partials::NTuple{P, A}) where {P, A <: AbstractArray}
        T = eltype(value)
        N = ndims(value)
        can_taylorize(T) || throw_cannot_taylorize(T)
        new{T, N, A, P}(value, partials)
    end
end

function TaylorArray{P}(value::A) where {A <: AbstractArray, P}
    TaylorArray(value, ntuple(i -> zeros(eltype(value), size(value)), Val(P)))
end

function TaylorArray{P}(value::A, seed::A) where {A <: AbstractArray, P}
    TaylorArray(
        value, ntuple(i -> i == 1 ? seed : zeros(eltype(value), size(value)), Val(P)))
end

# Indexing

Base.@propagate_inbounds function Base.getindex(a::TaylorArray, i::Int...)
    new_value = value(a)[i...]
    new_partials = map(p -> p[i...], partials(a))
    return TaylorScalar(new_value, new_partials)
end

Base.@propagate_inbounds function Base.setindex!(
        a::TaylorArray, s::TaylorScalar, i::Int...)
    value(a)[i...] = value(s)
    for j in 1:length(partials(a))
        partials(a)[j][i...] = partials(s)[j]
    end
    return a
end

Base.@propagate_inbounds function Base.setindex!(
        a::TaylorArray, s::Real, i::Int...)
    value(a)[i...] = s
    return a
end

# Invariant
for op in Symbol[:size, :eachindex, :IndexStyle]
    @eval Base.$(op)(x::TaylorArray) = Base.$(op)(value(x))
end
