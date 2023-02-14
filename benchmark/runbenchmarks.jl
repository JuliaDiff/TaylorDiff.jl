using Pkg
Pkg.develop(path = ".")
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using Dates
using HTTP: put

include("helper.jl")

context = provide_context(Buildkite())
results = benchmarkpkg(TaylorDiff)
(; benchmarkgroup, benchmarkconfig) = results
reconstructed = Dict("context" => context,
                     "suite" => benchmarkgroup,
                     "config" => benchmarkconfig)
put("https://benchmark.tansongchen.com/TaylorDiff.jl"; body = json(reconstructed))
