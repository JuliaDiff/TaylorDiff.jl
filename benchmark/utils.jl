import JSON: lower, json
using HTTP: put
using BenchmarkTools
using BenchmarkTools: Trial, TrialEstimate

function lower(results::BenchmarkResults)
    Dict("name" => results.name,
         "commit" => results.commit,
         "benchmarkgroup" => lower(results.benchmarkgroup),
         "date" => results.date,
         "julia_commit" => results.julia_commit,
         "vinfo" => results.vinfo,
         "benchmarkconfig" => results.benchmarkconfig)
end

lower(group::BenchmarkGroup) = Dict(key => lower(value) for (key, value) in group.data)

lower(trial::Trial) = lower(median(trial))

function lower(te::TrialEstimate)
    Dict("time" => te.time,
         "memory" => te.memory,
         "allocs" => te.allocs)
end

function upload(results::BenchmarkResults)
    put("https://benchmark.tansongchen.com/data?repo=TaylorDiff.jl"; body=json(results))
end
