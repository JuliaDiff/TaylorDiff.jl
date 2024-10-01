using ChainRules
using ChainRulesCore
using Symbolics: @variables
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: Pow

dummy = (NoTangent(), 1)
@variables z

function define_unary_function(func, m)
    F = typeof(func)
    # base case
    @eval m function (op::$F)(t::TaylorScalar{T, 1}) where {T}
        t0 = value(t)
        t1 = first(partials(t))
        f0, f1 = frule((NoTangent(), t1), op, t0)
        TaylorScalar{T, 1}(f0, zero_tangent(f0) + f1)
    end
    der = frule(dummy, func, z)[2]
    term, raiser = der isa Pow && der.exp == -1 ? (der.base, raiseinv) : (der, raise)
    # recursion by raising
    @eval m @generated function (op::$F)(t::TaylorScalar{T, N}) where {T, N}
        der_expr = $(QuoteNode(toexpr(term)))
        f = $func
        quote
            $(Expr(:meta, :inline))
            z = TaylorScalar{T, N - 1}(t)
            f0 = $f(value(t)[1])
            df = zero_tangent(z) + $der_expr
            $$raiser(f0, df, t)
        end
    end
end
