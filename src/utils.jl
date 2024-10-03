using ChainRules
using ChainRulesCore
using Symbolics: @variables, @rule, unwrap, isdiv
using SymbolicUtils.Code: toexpr

"""
Pick a strategy for raising the derivative of a function. If the derivative is like 1 over something, raise with the division rule; otherwise, raise with the multiplication rule.
"""
function get_term_raiser(func)
    @variables z
    r1 = @rule -1 * (1 / ~x) => (-1) / ~x
    der = frule((NoTangent(), true), func, z)[2]
    term = unwrap(der)
    maybe_rewrite = r1(term)
    if maybe_rewrite !== nothing
        term = maybe_rewrite
    end
    if isdiv(term) && (term.num == 1 || term.num == -1)
        term.den * term.num, raiseinv
    else
        term, raise
    end
end

function define_unary_function(func, m)
    F = typeof(func)
    # First order: call frule directly
    @eval m function (op::$F)(t::TaylorScalar{T, 1}) where {T}
        t0 = value(t)
        t1 = first(partials(t))
        f0, f1 = frule((NoTangent(), t1), op, t0)
        TaylorScalar{1}(T(f0), zero(T) + f1)
    end
    term, raiser = get_term_raiser(func)
    # Higher order: recursion by raising
    @eval m @generated function (op::$F)(t::TaylorScalar{T, p}) where {T, p}
        expr = $(QuoteNode(toexpr(term)))
        f = $func
        quote
            $(Expr(:meta, :inline))
            z = TaylorScalar{p - 1}(t)
            f0 = $f(value(t)[1])
            df = zero_tangent(z) + $expr
            $$raiser(f0, df, t)
        end
    end
end
