using Pkg
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using Dates
using HTTP: put

include("helper.jl")

context = provide_context(Dummy())
results = benchmarkpkg(TaylorDiff)
(; benchmarkgroup, benchmarkconfig) = results
reconstructed = Dict("context" => context,
    "suite" => benchmarkgroup,
    "config" => benchmarkconfig)
put("https://benchmark-data.tansongchen.workers.dev/TaylorDiff.jl";
    body = json(reconstructed))
