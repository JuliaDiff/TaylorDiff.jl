module TaylorDiffSFExt
using TaylorDiff, SpecialFunctions
using Symbolics: @variables
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: Pow
using TaylorDiff: value, raise
using ChainRules, ChainRulesCore

dummy = (NoTangent(), 1)
@variables z
for func in (erf,)
    F = typeof(func)
    # base case
    @eval function (op::$F)(t::TaylorScalar{T, 2}) where {T}
        t0, t1 = value(t)
        TaylorScalar{T, 2}(frule((NoTangent(), t1), op, t0))
    end
    der = frule(dummy, func, z)[2]
    term, raiser = der isa Pow && der.exp == -1 ? (der.base, raiseinv) : (der, raise)
    # recursion by raising
    @eval @generated function (op::$F)(t::TaylorScalar{T, N}) where {T, N}
        der_expr = $(QuoteNode(toexpr(term)))
        f = $func
        quote
            $(Expr(:meta, :inline))
            z = TaylorScalar{T, N - 1}(t)
            df = $der_expr
            $$raiser($f(value(t)[1]), df, t)
        end
    end
end

end
