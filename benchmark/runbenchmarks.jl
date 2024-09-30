using Pkg
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using Dates
using HTTP: put

struct BenchmarkContext
    datetime::DateTime
    commit::String
    branch::String
    tag::String
end

dict(x) = Dict(name => lower(getfield(x, name)) for name in fieldnames(typeof(x)))

lower(results::BenchmarkResults) = dict(results)
function lower(group::BenchmarkGroup)
    Dict(:tags => group.tags,
        :data => Dict(key => lower(value) for (key, value) in group.data))
end
lower(trial::Trial) = lower(minimum(trial))
lower(estimate::TrialEstimate) = dict(estimate)
lower(parameters::Parameters) = dict(parameters)
lower(context::BenchmarkContext) = dict(context)

getenv(name::String) = String(strip(ENV[name]))

context = if "BUILDKITE" in keys(ENV)
    BenchmarkContext(now(), # datetime
        getenv("BUILDKITE_COMMIT"), # commit
        getenv("BUILDKITE_BRANCH"), # branch
        getenv("BUILDKITE_TAG"))
else
    BenchmarkContext(now(), "abcdef123456", "dummy", "dummy")
end
display(context)

results = benchmarkpkg(TaylorDiff)
(; benchmarkgroup, benchmarkconfig) = results
reconstructed = Dict("context" => context,
    "suite" => benchmarkgroup,
    "config" => benchmarkconfig)
put("https://benchmark-data.tansongchen.workers.dev/TaylorDiff.jl";
    body = json(reconstructed))
