using Symbolics, SymbolicUtils

@variables x
p = Symbolics.variables(:p, 0:20)

struct A{T, P} <: Real
    a::T
    b::NTuple{P, T}
end

function make_nested_expressions(order)
    exprs = [x]
    for i in 1:order
        term = p[i + 1]
        for j in i:-1:1
            term += p[j] * exprs[i - j + 1]
        end
        push!(exprs, term)
    end
    exprs
end

for order in 1:17
    final_expr = make_nested_expressions(order)[end]
    print("Order: $order")
    build_function(final_expr, x, p; expression = Val(false), cse = true)
    @time build_function(final_expr, x, p; expression = Val(false), cse = true)
end

for order in 1:17
    exprs = make_nested_expressions(order)
    tuple_expr = Symbolics.Code.MakeTuple(Symbolics.unwrap.(exprs))
    print("Order: $order")
    @time f = build_function(tuple_expr, x, p; expression = Val(false), cse = true)
    @show f(1.0, 1.0:5.0)
end

# But it still doesn't scale with the unwrap trick. What else can I do?
for order in 1:17
    exprs = make_nested_expressions(order)
    struct_expr = term(A, Symbolics.unwrap(exprs[1]), (Symbolics.unwrap.(exprs[2:end])...,))
    print("Order: $order")
    @time f = build_function(struct_expr, x, p; expression = Val(false), cse = true)
    @show f(1.0, 1.0:5.0)
end
