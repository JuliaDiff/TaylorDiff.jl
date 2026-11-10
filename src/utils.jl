# This file is a bunch of compiler magics to cleverly define pushforward rules.
# If you are only interested in data structures and pushforward rules, you can skip this file.

using ChainRules
using ChainRulesCore
import DiffRules
using MacroTools
using MacroTools: prewalk, postwalk, @capture

"""
    get_term_raiser(func)

Return the derivative of `func` as an expression in `z`, together with the strategy for raising
it to higher orders: if the derivative is `inv(term)` or `1 / term` (possibly negated), `term`
(respectively `-term`) is raised with the division rule `raiseinv`; otherwise the
derivative is raised with the multiplication rule `raise`.

The expression is evaluated on Taylor series, so it must equal the derivative as a function
(not only numerically) and must not branch on values. Such expressions are taken from DiffRules,
whose rules are value-free expressions by design (unlike ChainRules' rules, which are only
meant to be evaluated on numbers).
"""
function get_term_raiser(func)
    f = nameof(func)
    M = DiffRules.hasdiffrule(:Base, f, 1) ? :Base : nameof(parentmodule(func))
    der = DiffRules.diffrule(M, f, :z)
    @capture(der, inv(den_) | 1 / den_) && return den, raiseinv
    @capture(der, -inv(den_) | -(1 / den_) | -1 / den_) && return :(-$den), raiseinv
    return der, raise
end
# DiffRules' rule for `abs` branches on the sign of its argument, and it has no rule for `sign`
get_term_raiser(::typeof(abs)) = :(sign(z)), raise
get_term_raiser(::typeof(sign)) = :(zero(z)), raise

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
        expr = $(QuoteNode(term))
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

tuplen(::Type{NTuple{N, T}}) where {N, T} = N
function interpolate(ex::Expr, dict)
    func = ex.args[1]
    args = map(x -> interpolate(x, dict), ex.args[2:end])
    getproperty(Base, func)(args...)
end
interpolate(ex::Symbol, dict) = get(dict, ex, ex)
interpolate(ex::Any, _) = ex

function unroll_loop(start, stop, var, body, d)
    ex = Expr(:block)
    start = interpolate(start, d)
    stop = interpolate(stop, d)
    for i in start:stop
        iter = prewalk(x -> x === var ? i : x, body)
        args = filter(x -> !(x isa LineNumberNode), iter.args)
        append!(ex.args, args)
    end
    ex
end

function process(d, expr)
    # Unroll loops
    expr = prewalk(expr) do x
        @match x begin
            for var_ in start_:stop_
                body_
            end => unroll_loop(start, stop, var, body, d)
            _ => x
        end
    end
    # Modify indices
    magic_names = (:v, :s, :c)
    known_names = Set()
    expr = postwalk(expr) do x
        @match x begin
            a_[idx_] => a in magic_names ? Symbol(a, idx) : :($a[begin + $idx])
            (a_ = b_) => (push!(known_names, a); :($a = $b))
            (a_ += b_) => a in known_names ? :($a += $b) :
                          (push!(known_names, a); :($a = $b))
            (a_ -= b_) => a in known_names ? :($a -= $b) :
                          (push!(known_names, a); :($a = -$b))
            TaylorScalar(v_) => :(TaylorScalar(tuple($([Symbol(v, idx) for idx in 0:d[:P]]...))))
            _ => x
        end
    end
    # Add inline meta
    return quote
        $(Expr(:meta, :inline))
        $expr
    end
end

"""
    immutable(def)

Transform a function definition to a @generated function.

1. Allocations are removed by replacing the output with scalar variables;
2. Loops are unrolled;
3. Indices are modified to use 1-based indexing;
"""
macro immutable(def)
    dict = splitdef(def)
    pairs = Any[]
    for symbol in dict[:whereparams]
        push!(pairs, :($(QuoteNode(symbol)) => $symbol))
    end
    esc(quote
        @generated function $(dict[:name])($(dict[:args]...)) where {$(dict[:whereparams]...)}
            d = Dict($(pairs...))
            process(d, $(QuoteNode(dict[:body])))
        end
    end)
end
