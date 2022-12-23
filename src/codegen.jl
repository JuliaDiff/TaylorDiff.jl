using ChainRulesCore
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: Pow

@scalar_rule +(x::Any) true
@scalar_rule -(x::Any) -1
@scalar_rule deg2rad(x::Any) deg2rad(one(x))
@scalar_rule rad2deg(x::Any) rad2deg(one(x))
@scalar_rule asin(x::Any) inv(sqrt(1 - x^2))
@scalar_rule acos(x::Any) inv(-sqrt(1 - x^2))
@scalar_rule atan(x::Any) inv(1 + x^2)
@scalar_rule log(x::Any) inv(x)
@scalar_rule log10(x::Any) inv(log(10.0) * x)
@scalar_rule log1p(x::Any) inv(x + 1)
@scalar_rule log2(x::Any) inv(log(2.0) * x)

dummy = (NoTangent(), 1)
@syms t₁
for func in (+, -, deg2rad, rad2deg, asin, acos, atan, log, log10, log1p, log2)
    F = typeof(func)
    # base case
    @eval function (op::$F)(t::TaylorScalar{T, 2}) where {T}
        t0, t1 = value(t)
        TaylorScalar{T, 2}(frule((NoTangent(), t1), op, t0))
    end
    der = frule(dummy, func, t₁)[2]
    term, raiser = der isa Pow && der.exp == -1 ? (der.base, raiseinv) : (der, raise)
    # recursion by raising
    @eval @generated function (op::$F)(t::TaylorScalar{T, N}) where {T, N}
        der_expr = $(QuoteNode(toexpr(term)))
        f = $func
        quote
            $(Expr(:meta, :inline))
            t₁ = TaylorScalar{T, N - 1}(t)
            df = $der_expr
            $$raiser($f(value(t)[1]), df, t)
        end
    end
end
