function create_benchmark_mlp(mlp_conf::Tuple{Int, Int}, x::Vector{T},
        l::Vector{T}) where {T <: Number}
    input, hidden = mlp_conf
    W₁, W₂, b₁, b₂ = rand(hidden, input), rand(1, hidden), rand(hidden), rand(1)
    σ = exp
    mlp(x) = first(W₂ * σ.(W₁ * x + b₁) + b₂)
    f1 = z -> ForwardDiff.derivative(t -> mlp(x + t * l), z)
    f2 = x -> ForwardDiff.derivative(f1, x)
    f3 = x -> ForwardDiff.derivative(f2, x)
    f4 = x -> ForwardDiff.derivative(f3, x)
    f5 = x -> ForwardDiff.derivative(f4, x)
    f6 = x -> ForwardDiff.derivative(f5, x)
    f7 = x -> ForwardDiff.derivative(f6, x)
    functions = Function[f1, f2, f3, f4, f5, f6, f7]
    forwarddiff, taylordiff = BenchmarkGroup(), BenchmarkGroup()
    for (index, func) in enumerate(functions)
        forwarddiff[index] = @benchmarkable $func(0)
    end
    Ns = [Val(order) for order in 1:7]
    for (index, N) in enumerate(Ns)
        taylordiff[index] = @benchmarkable derivative($mlp, $x, $l, $N)
    end
    return BenchmarkGroup(["vector"],
        "forwarddiff" => forwarddiff,
        "taylordiff" => taylordiff)
end
