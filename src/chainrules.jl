import ChainRulesCore: rrule, RuleConfig, ProjectTo, backing
using Zygote: @adjoint

function contract(a::TaylorScalar{T, N}, b::TaylorScalar{S, N}) where {T, S, N}
    mapreduce(*, +, value(a), value(b))
end

function rrule(::Type{TaylorScalar{T, N}}, v::NTuple{N, T}) where {N, T <: Number}
    taylor_scalar_pullback(t̄) = NoTangent(), value(t̄)
    return TaylorScalar(v), taylor_scalar_pullback
end

function rrule(::typeof(value), t::TaylorScalar{T, N}) where {N, T}
    value_pullback(v̄::NTuple{N, T}) = NoTangent(), TaylorScalar(v̄)
    # for structural tangent, convert to tuple
    function value_pullback(v̄::Tangent{P, NTuple{N, T}}) where {P}
        NoTangent(), TaylorScalar{T, N}(backing(v̄))
    end
    value_pullback(v̄) = NoTangent(), TaylorScalar{T, N}(map(x -> convert(T, x), Tuple(v̄)))
    return value(t), value_pullback
end

function rrule(::typeof(extract_derivative), t::TaylorScalar{T, N},
               i::Integer) where {N, T <: Number}
    function extract_derivative_pullback(d̄)
        NoTangent(), TaylorScalar{T, N}(ntuple(j -> j === i ? d̄ : zero(T), Val(N))),
        NoTangent()
    end
    return extract_derivative(t, i), extract_derivative_pullback
end

function rrule(::typeof(*), A::Matrix{S},
               t::Vector{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = ProjectTo(A)
    function gemv_pullback(x̄)
        NoTangent(), @thunk(project_A(contract.(x̄, transpose(t)))), @thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

@adjoint function +(t::Vector{TaylorScalar{T, N}}, v::Vector{T}) where {N, T <: Number}
    project_v = ProjectTo(v)
    t + v, x̄ -> (x̄, project_v(x̄))
end

@adjoint function +(v::Vector{T}, t::Vector{TaylorScalar{T, N}}) where {N, T <: Number}
    project_v = ProjectTo(v)
    v + t, x̄ -> (project_v(x̄), x̄)
end

(project::ProjectTo{T})(dx::TaylorScalar{T, N}) where {N, T <: Number} = primal(dx)

# Not-a-number patches

ProjectTo(::T) where {T <: TaylorScalar} = ProjectTo{T}()
(p::ProjectTo{T})(x::T) where {T <: TaylorScalar} = x
ProjectTo(x::AbstractArray{T}) where {T <: TaylorScalar} = ProjectTo{AbstractArray}(; element=ProjectTo(zero(T)), axes=axes(x))
(p::ProjectTo{AbstractArray{T}})(x::AbstractArray{T}) where {T <: TaylorScalar} = x
