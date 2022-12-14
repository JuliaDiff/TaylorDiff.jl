using ChainRulesCore
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: Pow

@scalar_rule asin(x::Any) inv(sqrt(1 - x ^ 2))
@scalar_rule acos(x::Any) -(inv(sqrt(1 - x ^ 2)))
@scalar_rule atan(x::Any) inv(1 + x ^ 2)

functions = Function[asin, acos, atan]
dummy = (NoTangent(), 1)
@syms t₁
for func in functions
    F = typeof(func)
    der = frule(dummy, func, t₁)[2]
    # if der isa Pow && der.exp == -1
    #     invder = der.base
    # end
    @eval @generated function (op::$F)(t::TaylorScalar{T,N}) where {T,N}
        der_expr = $(QuoteNode(toexpr(der)))
        f = $(func)
        quote
            $(Expr(:meta, :inline))
            t₁ = TaylorScalar{T,N-1}(t)
            df = $der_expr
            TaylorDiff.raise($f(value(t)[1]), df, t)
        end
    end
end
