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

abstract type ContextProvider end

struct Dummy <: ContextProvider end
provide_context(::Dummy) = BenchmarkContext(now(), "abcdef123456", "dummy", "dummy")

struct Buildkite <: ContextProvider end
function provide_context(::Buildkite)
    getenv(name::String) = String(strip(ENV[name]))
    BenchmarkContext(now(), # datetime
                     getenv("BUILDKITE_COMMIT"), # commit
                     getenv("BUILDKITE_BRANCH"), # branch
                     getenv("BUILDKITE_TAG"))
end
