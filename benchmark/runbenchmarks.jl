using TaylorDiff
using BenchmarkTools, PkgBenchmark
using BenchmarkTools: Trial, TrialEstimate, Parameters
import JSON: lower, json
using HTTP: put
using LibGit2: GitRepo, headname

dict(x) = Dict(name => lower(getfield(x, name)) for name in fieldnames(typeof(x)))

lower(results::BenchmarkResults) = dict(results)
lower(group::BenchmarkGroup) = Dict(key => lower(value) for (key, value) in group.data)
lower(trial::Trial) = lower(median(trial))
lower(estimate::TrialEstimate) = dict(estimate)
lower(parameters::Parameters) = dict(parameters)

function benchmark()
    repo = GitRepo(pwd())
    branch = headname(repo)
    config = BenchmarkConfig(id = branch)
    results = benchmarkpkg(TaylorDiff, config)
    endpoint = "https://benchmark.tansongchen.com"
    put(endpoint; body = json(results))
end

benchmark()
