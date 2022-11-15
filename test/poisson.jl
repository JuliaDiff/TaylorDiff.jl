using Flux, Zygote

network = Chain(
    Dense(2, 16, relu),
    Dense(16, 16, relu),
    Dense(16, 1),
    first
)
f(x) = network(x) * x[1] * (1 - x[1]) * x[2] * (1 - x[2])

using ForwardDiff

dataset = [rand(2) for i in 1:500]

function loss(;n=100)
    out = 0.0
    for i in 1:n
        x = dataset[i]
        v = ForwardDiff.gradient(f, x)
        out += abs2(v'v + sin(pi*x[1])*sin(pi*x[2]))
    end
    out
end

iter = 0
cb = function ()
    global iter
    iter += 1
    if iter%10 == 0
        println(loss())
    end
end

data = Iterators.repeated((), 1000)
Flux.train!(loss, Flux.params(network), data, Flux.ADATaylor(0.05),cb=cb)

# grid = 0:0.01:1
# analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)
# u = [NNfull([x,y]) for x in grid, y in grid]
# utrue = [analytic_sol_func(x,y) for x in grid, y in grid]
# diff_u = abs.(u .- utrue)
