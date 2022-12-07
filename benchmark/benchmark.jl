using TaylorDiff
using ForwardDiff
using Lux
using BenchmarkTools

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
    nested, taylor = Float64[], Float64[]
    for func in functions
        t = @benchmark $func($(Ref(x))[])
        push!(nested, median(t).time)
    end
    orders = 1:10
    Ns = [Val{order + 1}() for order in orders]
    for N in Ns
        t = @benchmark derivative($f, $x, $N)
        push!(taylor, median(t).time)
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
        println(estim)
    end
    orders = 1:10
    Ns = [Val{order + 1}() for order in orders]
    for N in Ns
        trial = @benchmark derivative($f, $x, $l, $N) samples=10
        estim = median(trial)
        println(estim)
    end
    return nested, taylor
end
