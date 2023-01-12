using Flux
using ChainRulesCore: @opt_out
using TaylorDiff
using Zygote
using Plots

const input = 2
const hidden = 16

model = Chain(
    Dense(input => hidden, sin),
    Dense(hidden => hidden, sin),
    Dense(hidden => 1),
    first
)
trial(model, x) = model(x)

ε = cbrt(eps(Float32))
ε₁ = [ε, 0]
ε₂ = [0, ε]

M = 100
data = [rand(input) for _ in 1:M]
function loss_by_finitediff(model, x)
    error = (trial(model, x + ε₁) + trial(model, x - ε₁) + trial(model, x + ε₂) +
             trial(model, x - ε₂) - 4 * trial(model, x)) /
            ε^2 + sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end
function loss_by_taylordiff(model, x)
    f(x) = trial(model, x)
    error = derivative(f, x, [1., 0.], 2) + derivative(f, x, [0., 1.], 2) + sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end

opt = Flux.setup(Adam(), model)

allloss(model, loss) = sum([loss(model, x) for x in data])
for epoch in 1:1000
    Flux.train!(loss_by_taylordiff, model, data, opt)
end

grid = 0:0.01:1
solution(x, y) = (sin(π * x) * sin(π * y)) / (2π^2)
u = [trial(model, [x, y]) for x in grid, y in grid]
utrue = [solution(x, y) for x in grid, y in grid]
diff_u = abs.(u .- utrue)

surface(u)
surface(utrue)
surface(diff_u)
