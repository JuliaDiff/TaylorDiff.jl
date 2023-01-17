using BenchmarkTools
using Random: seed!
using ForwardDiff, Zygote, Flux
using TaylorSeries: Taylor1
using TaylorDiff

seed!(19260817)

include("scalar.jl")
include("mlp.jl")
include("taylor_expansion.jl")
include("pinn.jl")

scalar = create_benchmark_scalar_function(sin, 0.1)
mlp = create_benchmark_mlp((2, 16), [2.0, 3.0], [1.0, 1.0])

const SUITE = BenchmarkGroup("scalar" => scalar,
                             "mlp" => mlp,
                             "taylor_expansion" => taylor_expansion,
                             "pinn" => pinn)
