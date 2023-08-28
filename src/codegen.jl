using ChainRulesCore
using SpecialFunctions
using SymbolicUtils, SymbolicUtils.Code
using SymbolicUtils: BasicSymbolic, Pow

@scalar_rule +(x::BasicSymbolic) true
@scalar_rule -(x::BasicSymbolic) -1
@scalar_rule deg2rad(x::BasicSymbolic) deg2rad(one(x))
@scalar_rule rad2deg(x::BasicSymbolic) rad2deg(one(x))
@scalar_rule asin(x::BasicSymbolic) inv(sqrt(1 - x^2))
@scalar_rule acos(x::BasicSymbolic) inv(-sqrt(1 - x^2))
@scalar_rule atan(x::BasicSymbolic) inv(-(1 + x^2))
@scalar_rule acot(x::BasicSymbolic) inv(-(1 + x^2))
@scalar_rule acsc(x::BasicSymbolic) inv(x^2 * -sqrt(1 - x^-2))
@scalar_rule asec(x::BasicSymbolic) inv(x^2 * sqrt(1 - x^-2))
@scalar_rule log(x::BasicSymbolic) inv(x)
@scalar_rule log10(x::BasicSymbolic) inv(log(10.0) * x)
@scalar_rule log1p(x::BasicSymbolic) inv(x + 1)
@scalar_rule log2(x::BasicSymbolic) inv(log(2.0) * x)
@scalar_rule sinh(x::BasicSymbolic) cosh(x)
@scalar_rule cosh(x::BasicSymbolic) sinh(x)
@scalar_rule tanh(x::BasicSymbolic) 1-Ω^2
@scalar_rule acosh(x::BasicSymbolic) inv(sqrt(x - 1) * sqrt(x + 1))
@scalar_rule acoth(x::BasicSymbolic) inv(1 - x^2)
@scalar_rule acsch(x::BasicSymbolic) inv(x^2 * -sqrt(1 + x^-2))
@scalar_rule asech(x::BasicSymbolic) inv(x * -sqrt(1 - x^2))
@scalar_rule asinh(x::BasicSymbolic) inv(sqrt(x^2 + 1))
@scalar_rule atanh(x::BasicSymbolic) inv(1 - x^2)
@scalar_rule erf(x::BasicSymbolic) exp(-x^2)*(2 / sqrt(pi))

dummy = (NoTangent(), 1)
@syms t₁
for func in (+, -, deg2rad, rad2deg,
    sinh, cosh, tanh,
    asin, acos, atan, asec, acsc, acot,
    log, log10, log1p, log2,
    asinh, acosh, atanh, asech, acsch,
    acoth, erf)
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
