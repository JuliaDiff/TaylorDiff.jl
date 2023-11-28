using ChainRules
using ChainRulesCore
using Symbolics: @variables
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: Pow

dummy = (NoTangent(), 1)
@variables z
for func in (+, -, deg2rad, rad2deg,
    sinh, cosh, tanh,
    asin, acos, atan, asec, acsc, acot,
    log, log10, log1p, log2,
    asinh, acosh, atanh, asech, acsch,
    acoth,
    abs, sign)
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
