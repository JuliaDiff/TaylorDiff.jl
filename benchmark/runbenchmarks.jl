using Pkg
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using Dates
using HTTP: put
using Random

dict(x) = Dict(name => lower(getfield(x, name)) for name in fieldnames(typeof(x)))
lower(results::BenchmarkResults) = dict(results)
function lower(group::BenchmarkGroup)
    Dict(:tags => group.tags,
        :data => [Dict(lower(value)..., "name" => key) for (key, value) in group.data])
end
lower(trial::Trial) = lower(minimum(trial))
lower(estimate::TrialEstimate) = dict(estimate)
lower(parameters::Parameters) = dict(parameters)

getenv(name::String) = String(strip(ENV[name]))

body = Dict("name" => "TaylorDiff.jl", "datetime" => now())
if "BUILDKITE" in keys(ENV)
    body["commit"] = getenv("BUILDKITE_COMMIT")
    body["branch"] = getenv("BUILDKITE_BRANCH")
    body["tag"] = getenv("BUILDKITE_TAG")
else
    body["commit"] = randstring("abcdef0123456789", 40)
    body["branch"] = "dummy"
end

(; benchmarkgroup, benchmarkconfig) = benchmarkpkg(TaylorDiff)
body["config"] = benchmarkconfig
body["result"] = lower(benchmarkgroup)[:data]
put("https://benchmark-data.tansongchen.workers.dev"; body = json(body))
