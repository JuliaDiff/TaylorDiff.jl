function create_benchmark_scalar_function(f::F, x::T) where {F, T <: Number}
    f1 = x -> ForwardDiff.derivative(f, x)
    f2 = x -> ForwardDiff.derivative(f1, x)
    f3 = x -> ForwardDiff.derivative(f2, x)
    f4 = x -> ForwardDiff.derivative(f3, x)
    f5 = x -> ForwardDiff.derivative(f4, x)
    f6 = x -> ForwardDiff.derivative(f5, x)
    f7 = x -> ForwardDiff.derivative(f6, x)
    f8 = x -> ForwardDiff.derivative(f7, x)
    f9 = x -> ForwardDiff.derivative(f8, x)
    functions = Function[f1, f2, f3, f4, f5, f6, f7, f8, f9]
    forwarddiff_group = BenchmarkGroup([index => @benchmarkable $func($(Ref(x))[])
                                        for (index, func) in enumerate(functions)]...)
    taylordiff_group = BenchmarkGroup()
    Ns = [Val(order) for order in 1:9]
    for (index, N) in enumerate(Ns)
        taylordiff_group[index] = @benchmarkable derivative($f, $x, one($x), $N)
    end
    return BenchmarkGroup(["scalar"],
        "forwarddiff" => forwarddiff_group,
        "taylordiff" => taylordiff_group)
end
