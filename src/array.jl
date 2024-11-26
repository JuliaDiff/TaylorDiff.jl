export TaylorArray

"""
    TaylorArray{T, N, A, P}

Representation of Taylor polynomials in array mode.

# Fields

- `value::A`: zeroth order coefficient
- `partials::NTuple{P, A}`: i-th element of this stores the i-th derivative
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

TaylorArray(all::NTuple{P, A}) where {A, P} = TaylorArray(all[1], all[2:end])

function TaylorArray{P}(value::A) where {A <: AbstractArray, P}
    TaylorArray(value, ntuple(i -> broadcast(zero, value), Val(P)))
end

function TaylorArray{P}(value::A, seed::A) where {A <: AbstractArray, P}
    TaylorArray(
        value, ntuple(i -> i == 1 ? seed : broadcast(zero, seed), Val(P)))
end

# Necessary AbstractArray interface methods for TaylorArray to work
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

## 1. Invariant
for op in Symbol[:size, :strides, :eachindex, :IndexStyle]
    @eval Base.$(op)(x::TaylorArray) = Base.$(op)(value(x))
end

## 2. Indexing
function Base.similar(a::TaylorArray, ::Type{<:TaylorScalar{T}}, dims::Dims) where {T}
    new_value = similar(value(a), T, dims)
    new_partials = map(p -> similar(p, T, dims), partials(a))
    return TaylorArray(new_value, new_partials)
end

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

## 3. Broadcasting
struct TaylorArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
TaylorArrayStyle(::Val{N}) where {N} = TaylorArrayStyle{N}()
TaylorArrayStyle{M}(::Val{N}) where {N, M} = TaylorArrayStyle{N}()

Base.BroadcastStyle(::Type{<:TaylorArray{T, N}}) where {T, N} = TaylorArrayStyle{N}()
# This is added to make Zygote custom broadcasting work
# However, we might implement custom broadcasting semantics for TaylorArray in the future
# function Base.BroadcastStyle(::Type{<:Array{
#         <:Tuple{TaylorScalar{T, P}, Any}, N}}) where {T, N, P}
#     TaylorArrayStyle{N}()
# end

function Base.similar(
        bc::Broadcast.Broadcasted{<:TaylorArrayStyle}, ::Type{ElType}) where {ElType}
    A = find_taylor(bc)
    similar(A, ElType, axes(bc))
end

find_taylor(bc::Broadcast.Broadcasted) = find_taylor(bc.args)
find_taylor(args::Tuple) = find_taylor(find_taylor(args[1]), Base.tail(args))
find_taylor(x) = x
find_taylor(::Tuple{}) = nothing
find_taylor(a::TaylorArray, rest) = a
function find_taylor(a::Array{<:Tuple{TaylorScalar{T, P}, Any}, N}, rest) where {T, P, N}
    TaylorArray{P}(zeros(T, size(a)))
end
find_taylor(::Any, rest) = find_taylor(rest)

# function Base.copyto!(dest::TaylorArray, bc::Broadcast.Broadcasted{<:TaylorArrayStyle, Axes}) where Axes
#     println("copyto!($(typeof(dest)), $(typeof(bc)))")
#     error("Not implemented")
# end
