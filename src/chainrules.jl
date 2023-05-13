import ChainRulesCore: rrule, RuleConfig, ProjectTo, backing
using Base.Broadcast: broadcasted
import Zygote: @adjoint, accum_sum, unbroadcast, Numeric, ∇getindex, _project

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

function rrule(::typeof(*), A::AbstractMatrix{S},
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
function ProjectTo(x::AbstractArray{T}) where {T <: TaylorScalar}
    ProjectTo{AbstractArray}(; element = ProjectTo(zero(T)), axes = axes(x))
end
(p::ProjectTo{AbstractArray{T}})(x::AbstractArray{T}) where {T <: TaylorScalar} = x
accum_sum(xs::AbstractArray{T}; dims = :) where {T <: TaylorScalar} = sum(xs, dims = dims)

TaylorNumeric{T <: TaylorScalar} = Union{T, AbstractArray{<:T}}

@adjoint function broadcasted(::typeof(+), xs::Union{Numeric, TaylorNumeric}...)
    broadcast(+, xs...), ȳ -> (nothing, map(x -> unbroadcast(x, ȳ), xs)...)
end

struct TaylorOneElement{T, N, I, A} <: AbstractArray{T, N}
    val::T
    ind::I
    axes::A
    function TaylorOneElement(val::T, ind::I,
                              axes::A) where {T <: TaylorScalar, I <: NTuple{N, Int},
                                              A <: NTuple{N, AbstractUnitRange}} where {N}
        new{T, N, I, A}(val, ind, axes)
    end
end

Base.size(A::TaylorOneElement) = map(length, A.axes)
Base.axes(A::TaylorOneElement) = A.axes
function Base.getindex(A::TaylorOneElement{T, N}, i::Vararg{Int, N}) where {T, N}
    ifelse(i == A.ind, A.val, zero(T))
end

function ∇getindex(x::AbstractArray{T, N}, inds) where {T <: TaylorScalar, N}
    dy -> begin
        dx = TaylorOneElement(dy, inds, axes(x))
        return (_project(x, dx), map(_ -> nothing, inds)...)
    end
end

@generated function mul_adjoint(Ω::TaylorScalar{T, N}, x::TaylorScalar{T, N}) where {T, N}
    return quote
        vΩ, vx = value(Ω), value(x)
        @inbounds TaylorScalar($([:(+($([:($(binomial(j - 1, i - 1)) * vΩ[$j] *
                                           vx[$(j + 1 - i)]) for j in i:N]...)))
                                  for i in 1:N]...))
    end
end

rrule(::typeof(*), x::TaylorScalar) = rrule(identity, x)

function rrule(::typeof(*), x::TaylorScalar, y::TaylorScalar)
    function times_pullback2(Ω̇)
        ΔΩ = unthunk(Ω̇)
        return (NoTangent(), ProjectTo(x)(mul_adjoint(ΔΩ, y)),
                ProjectTo(y)(mul_adjoint(ΔΩ, x)))
    end
    return x * y, times_pullback2
end

function rrule(::typeof(*), x::TaylorScalar, y::TaylorScalar, z::TaylorScalar,
               more::TaylorScalar...)
    Ω2, back2 = rrule(*, x, y)
    Ω3, back3 = rrule(*, Ω2, z)
    Ω4, back4 = rrule(*, Ω3, more...)
    function times_pullback4(Ω̇)
        Δ4 = back4(unthunk(Ω̇))  # (0, ΔΩ3, Δmore...)
        Δ3 = back3(Δ4[2])       # (0, ΔΩ2, Δz)
        Δ2 = back2(Δ3[2])       # (0, Δx, Δy)
        return (Δ2..., Δ3[3], Δ4[3:end]...)
    end
    return Ω4, times_pullback4
end
