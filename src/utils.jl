# This file is a bunch of compiler magics to cleverly define pushforward rules.
# If you are only interested in data structures and pushforward rules, you can skip this file.

using ChainRules
using ChainRulesCore
using Symbolics: @variables, @rule, unwrap, isdiv
using SymbolicUtils.Code: toexpr
using MacroTools
using MacroTools: prewalk, postwalk

"""
Pick a strategy for raising the derivative of a function.
If the derivative is like 1 over something, raise with the division rule;
otherwise, raise with the multiplication rule.
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
