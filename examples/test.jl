using Symbolics, SymbolicUtils
struct A1{T} <: Real
    x::T
    y::T
    function A1(value::T, partials::T) where {T}
        new{T}(value, partials)
    end
end

struct A2{T, P} <: Real
    x::T
    y::NTuple{P, T}
    function A2(value::T, partials::NTuple{P, T}) where {T, P}
        new{T, P}(value, partials)
    end
end

@register_symbolic A1(x, y)
@register_symbolic A2(x, y)
@variables x::Real
a1 = A1(x, x^2)
a2 = A2(x, (x^2, x^3))

fa1 = build_function(a1, x; expression = Val(false))
fa1(2.0)
fa2 = build_function(a2, x; expression = Val(false))
fa2(2.0)

function Symbolics.Code.toexpr(a2::A2, st)
    :($A2($(Symbolics.Code.toexpr(a2.x, st)),
        $(Symbolics.Code.toexpr(Symbolics.Code.MakeTuple(a2.y), st))))
end
fa2 = build_function(a2, x; expression = Val(false))
fa2(2.0)
