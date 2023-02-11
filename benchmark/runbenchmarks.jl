using Pkg
Pkg.develop(path = ".")
Pkg.instantiate()

using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using Dates
using HTTP: put

dict(x) = Dict(name => lower(getfield(x, name)) for name in fieldnames(typeof(x)))

lower(results::BenchmarkResults) = dict(results)
lower(group::BenchmarkGroup) = Dict(key => lower(value) for (key, value) in group.data)
lower(trial::Trial) = lower(median(trial))
lower(estimate::TrialEstimate) = dict(estimate)
lower(parameters::Parameters) = dict(parameters)

struct BenchmarkContext
    datetime::DateTime
    commit::String
    branch::String
    tag::String
end

lower(context::BenchmarkContext) = dict(context)

abstract type ContextProvider end

struct Dummy <: ContextProvider end
provide_context(::Dummy) = BenchmarkContext(now(), "", "", "")

struct Buildkite <: ContextProvider end
function provide_context(::Buildkite)
    getenv(name::String) = String(strip(ENV[name]))
    BenchmarkContext(now(), # datetime
                     getenv("BUILDKITE_COMMIT"), # commit
                     getenv("BUILDKITE_BRANCH"), # branch
                     getenv("BUILDKITE_TAG"))
end

context = provide_context(Buildkite())
results = benchmarkpkg(TaylorDiff)
(; benchmarkgroup, benchmarkconfig) = results
reconstructed = Dict("suite" => benchmarkgroup,
                     "context" => context,
                     "config" => benchmarkconfig)
put("https://benchmark.tansongchen.com/TaylorDiff.jl"; body = json(reconstructed))
