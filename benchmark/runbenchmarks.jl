using Pkg
Pkg.develop(path = ".")
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using HTTP: put

dict(x) = Dict(name => lower(getfield(x, name)) for name in fieldnames(typeof(x)))

lower(results::BenchmarkResults) = dict(results)
lower(group::BenchmarkGroup) = Dict(key => lower(value) for (key, value) in group.data)
lower(trial::Trial) = lower(median(trial))
lower(estimate::TrialEstimate) = dict(estimate)
lower(parameters::Parameters) = dict(parameters)

function benchmark()
    results = benchmarkpkg(TaylorDiff)
    endpoint = "https://benchmark.tansongchen.com"
    put(endpoint; body = json(results))
end

benchmark()
