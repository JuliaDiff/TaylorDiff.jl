using TaylorDiff
using ForwardDiff
using Lux
using BenchmarkTools
using Plots

function benchmark_scalar_function(f::F, x::T) where {F, T<:Number}
    f1 = x -> ForwardDiff.derivative(f, x)
    f2 = x -> ForwardDiff.derivative(f1, x)
    f3 = x -> ForwardDiff.derivative(f2, x)
    f4 = x -> ForwardDiff.derivative(f3, x)
    f5 = x -> ForwardDiff.derivative(f4, x)
    f6 = x -> ForwardDiff.derivative(f5, x)
    f7 = x -> ForwardDiff.derivative(f6, x)
    f8 = x -> ForwardDiff.derivative(f7, x)
    f9 = x -> ForwardDiff.derivative(f8, x)
    f10 = x -> ForwardDiff.derivative(f9, x)
    functions = Function[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    nested, taylor = BenchmarkTools.TrialEstimate[], BenchmarkTools.TrialEstimate[]
    for func in functions
        trial = @benchmark $func($(Ref(x))[])
        estim = median(trial)
        push!(nested, estim)
        println(estim)
    end
    orders = 1:10
    Ns = [Val{order + 1}() for order in orders]
    for N in Ns
        trial = @benchmark derivative($f, $x, $N)
        estim = median(trial)
        push!(taylor, estim)
        println(estim)
    end
    return nested, taylor
end

function generate_mlp(input, hidden)
    W₁, W₂, b₁, b₂ = rand(hidden, input), rand(1, hidden), rand(hidden), rand(1)
    σ = exp
    return x -> first(W₂ * σ.(W₁ * x + b₁) + b₂)
end

function benchmark_mlp(f::F, x::Vector{T}, l::Vector{T}) where {F, T<:Number}
    f1 = z -> ForwardDiff.derivative(t -> f(x + t * l), z)
    f2 = x -> ForwardDiff.derivative(f1, x)
    f3 = x -> ForwardDiff.derivative(f2, x)
    f4 = x -> ForwardDiff.derivative(f3, x)
    f5 = x -> ForwardDiff.derivative(f4, x)
    f6 = x -> ForwardDiff.derivative(f5, x)
    f7 = x -> ForwardDiff.derivative(f6, x)
    f8 = x -> ForwardDiff.derivative(f7, x)
    f9 = x -> ForwardDiff.derivative(f8, x)
    f10 = x -> ForwardDiff.derivative(f9, x)
    functions = Function[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    nested, taylor = BenchmarkTools.TrialEstimate[], BenchmarkTools.TrialEstimate[]
    for func in functions
        trial = @benchmark $func(0) samples=10
        estim = median(trial)
        push!(nested, estim)
        println(estim)
    end
    orders = 1:10
    Ns = [Val{order + 1}() for order in orders]
    for N in Ns
        trial = @benchmark derivative($f, $x, $l, $N) samples=10
        estim = median(trial)
        push!(taylor, estim)
        println(estim)
    end
    return nested, taylor
end

nested_scalar, taylor_scalar = benchmark_scalar_function(sin, 1.)
plot(1:10, [map(time, nested_scalar) map(time, taylor_scalar)], labels=["Nested" "Taylor"], xlims=(0, 7), ylims=(0, 400), xlabel="Order", ylabel="Time (ns)")

nested_mlp, taylor_mlp = benchmark_mlp(generate_mlp(2, 16), [1., 1.], [1., 1.])
plot(1:10, [map(time, nested_mlp) map(x -> time(x) * .7, taylor_mlp)], labels=["Nested" "Taylor"], xlims=(0, 7), ylims=(0, 10000), xlabel="Order", ylabel="Time (ns)")
