using BenchmarkTools
using Random: seed!, default_rng
using ForwardDiff, Zygote
using TaylorSeries: Taylor1
using TaylorDiff

rng = default_rng()
seed!(rng, 19260817)

using Logging
Logging.disable_logging(Logging.Warn)

include("groups/scalar.jl")
include("groups/mlp.jl")
include("groups/taylor_expansion.jl")
include("groups/pinn.jl")

scalar = create_benchmark_scalar_function(sin, 0.1)
mlp = create_benchmark_mlp((2, 16), [2.0, 3.0], [1.0, 1.0])

const SUITE = BenchmarkGroup("scalar" => scalar,
    "mlp" => mlp,
    "taylor_expansion" => taylor_expansion,
    "pinn" => pinn)
