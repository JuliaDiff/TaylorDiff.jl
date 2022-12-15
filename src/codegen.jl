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
    # base case
    @eval function (op::$F)(t::TaylorScalar{T,2}) where {T}
        t0, t1 = value(t)
        TaylorScalar{T,2}(frule((NoTangent(), t1), op, t0))
    end
    der = frule(dummy, func, t₁)[2]
    term, raiser = der isa Pow && der.exp == -1 ? (der.base, raiseinv) : (der, raise)
    # recursion by raising
    @eval @generated function (op::$F)(t::TaylorScalar{T,N}) where {T,N}
        der_expr = $(QuoteNode(toexpr(der)))
        f = $func
        quote
            $(Expr(:meta, :inline))
            t₁ = TaylorScalar{T,N-1}(t)
            df = $der_expr
            $$raiser($f(value(t)[1]), df, t)
        end
    end
end
