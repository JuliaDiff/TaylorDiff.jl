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
    sizes = [Val{order + 1}() for order in orders]
    for size in sizes
        t = @benchmark derivative($f, $x, $size)
        push!(taylor, median(t).time)
    end
    return nested, taylor
end
