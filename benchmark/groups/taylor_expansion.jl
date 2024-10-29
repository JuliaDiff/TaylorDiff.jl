function my_calculation(t, p, α, s)
    x = 1.0 / (1.0 - s * (t + 1) / (t - 1))
    rez = zero(x)
    for i in eachindex(p)
        rez += p[i] * x^α[i]
    end
    return rez * sqrt(2) / (1 - t)
end

N, m = 100, 20
p, α, s = rand(N), rand(N), rand()
p ./= sum(p)
t_ts = Taylor1(eltype(p), m)
t_td = TaylorScalar{m}(zero(eltype(p)), one(eltype(p)))
taylor_expansion = BenchmarkGroup(["scalar", "very-high-order"],
    "taylorseries" => (@benchmarkable my_calculation($t_ts,
        $p, $α,
        $s)),
    "taylordiff" => (@benchmarkable my_calculation($t_td, $p,
        $α, $s)))
