using Lux, Zygote, Enzyme, ComponentArrays

function trial(model, x, ps, st)
    u, st = Lux.apply(model, x, ps, st)
    x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * u
end

function loss_by_finitediff(model, x, ps, st)
    T = eltype(x)
    ε = cbrt(eps(T))
    ε₁ = [ε, zero(T)]
    ε₂ = [zero(T), ε]
    f(x) = trial(model, x, ps, st)
    error = (f(x + ε₁) + f(x - ε₁) + f(x + ε₂) + f(x - ε₂) - 4 * f(x)) / ε^2 +
            sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end
function loss_by_taylordiff(model, x, ps, st)
    f(x) = trial(model, x, ps, st)
    error = derivative(f, x, Float32[1, 0], Val(2)) +
            derivative(f, x, Float32[0, 1], Val(2)) +
            sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end
function loss_by_forwarddiff(model, x, ps, st)
    f(x) = trial(model, x, ps, st)
    error = derivative(f, x, Float32[1, 0], Val(2)) +
            derivative(f, x, Float32[0, 1], Val(2)) +
            sin(π * x[1]) * sin(π * x[2])
    abs2(error)
end

const input = 2
const hidden = 16
model = Chain(Dense(input => hidden, exp),
    Dense(hidden => hidden, exp),
    Dense(hidden => 1),
    first)
x = rand(Float32, input)
dx = deepcopy(x)
ps, st = Lux.setup(rng, model)
ps = ps |> ComponentArray
dps = deepcopy(ps)
dx .= 0;
dps .= 0;

pinn_t = BenchmarkGroup(
    "primal" => (@benchmarkable loss_by_taylordiff($model, $x, $ps, $st)),
    "gradient" => (@benchmarkable gradient(loss_by_taylordiff, $model,
        $x, $ps, $st)))
pinn_f = BenchmarkGroup(
    "primal" => (@benchmarkable loss_by_finitediff($model, $x, $ps, $st)),
    "gradient" => (@benchmarkable gradient($loss_by_finitediff, $model,
        $x, $ps, $st)))
pinn = BenchmarkGroup(["vector", "physical"], "taylordiff" => pinn_t,
    "finitediff" => pinn_f)
