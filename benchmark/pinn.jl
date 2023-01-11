using TaylorDiff, Zygote

const input = 2
const hidden = 16

struct PINN
    W₁
    b₁
    W₂
    b₂
end

(pinn::PINN)(x) = x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * first(pinn.W₂ * exp.(pinn.W₁ * x + pinn.b₁) + pinn.b₂)

dataset = [rand(input) for i in 1:10]
function loss(pinn)
    out = 0.0
    for x in dataset
        out += derivative(pinn, x, [1., 0.], Val(2))
    end
    out
end

myPINN = PINN(rand(hidden, input), rand(hidden), rand(1, hidden), rand(1))

gradient(loss, myPINN)
