using TaylorDiff, Zygote
using LinearAlgebra: dot

sigmoid(x) = (one(x) + tanh(x)) / 2

struct LinearModel
    c::Matrix
end

(m::LinearModel)(x) = sigmoid((m.c * x)[1])

data = rand(2)

function loss(model)
    derivative(model, data, [1., 0.], Val(2))
end

model = LinearModel(rand(1, 2))
gradient(loss, model)
