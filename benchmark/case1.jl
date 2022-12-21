using TaylorSeries, TaylorDiff, BenchmarkTools

function f(t, p, α, s)
    x = 1.0 / (1.0 - s * (t + 1) / (t - 1))
    rez = zero(x)
    for i in eachindex(p)
        rez += p[i] * x^α[i]
    end
    return rez * sqrt(2) / (1 - t)
end

# Random Parameters
N = 100
m = 20
p, α, s = (rand(N), rand(N), rand())
p ./= sum(p)

# Test
t0 = Taylor1(eltype(p), m)
t = TaylorScalar{Float64, 21}(0.0)
@btime f(t0, p, α, s);
@btime f(t, p, α, s);
