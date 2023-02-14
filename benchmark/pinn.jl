const input = 2
const hidden = 16

model = Chain(Dense(input => hidden, exp),
              Dense(hidden => hidden, exp),
              Dense(hidden => 1),
              first)
trial(model, x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * model(x)

x = rand(Float32, input)
function loss_by_finitediff(model, x)
    ε = cbrt(eps(Float32))
    ε₁ = [ε, 0]
    ε₂ = [0, ε]
    error = (trial(model, x + ε₁) + trial(model, x - ε₁) + trial(model, x + ε₂) +
             trial(model, x - ε₂) - 4 * trial(model, x)) /
            ε^2 + sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end
function loss_by_taylordiff(model, x)
    f(x) = trial(model, x)
    error = derivative(f, x, Float32[1, 0], Val(3)) +
            derivative(f, x, Float32[0, 1], Val(3)) +
            sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end

pinn_t = BenchmarkGroup("primal" => (@benchmarkable loss_by_taylordiff($model, $x)),
                        "gradient" => (@benchmarkable gradient(loss_by_taylordiff, $model,
                                                               $x)))
pinn_f = BenchmarkGroup("primal" => (@benchmarkable loss_by_finitediff($model, $x)),
                        "gradient" => (@benchmarkable gradient($loss_by_finitediff, $model,
                                                               $x)))
pinn = BenchmarkGroup(["vector", "physical"], "taylordiff" => pinn_t, "finitediff" => pinn_f)
