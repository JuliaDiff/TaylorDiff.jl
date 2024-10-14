using LinearAlgebra
import DifferentiationInterface
using DifferentiationInterface: AutoZygote, AutoEnzyme
import Zygote, Enzyme
using FiniteDiff: finite_difference_derivative

DI = DifferentiationInterface
backend = AutoZygote()
# backend = AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Const)

@testset "Zygote-over-TaylorDiff on same variable" begin
    # Scalar functions
    some_number = 0.7
    some_numbers = [0.3, 0.4, 0.1]
    for f in (exp, log, sqrt, sin, asin, sinh, asinh, x -> x^3)
        @test DI.derivative(x -> derivative(f, x, Val(2)), backend, some_number) ≈
              derivative(f, some_number, Val(3))
        @test DI.jacobian(x -> derivative.(f, x, Val(2)), backend, some_numbers) ≈
              diagm(derivative.(f, some_numbers, Val(3)))
    end

    # Vector functions
    g(x) = x[1] * x[1] + x[2] * x[2]
    @test DI.gradient(x -> derivative(g, x, [1.0, 0.0], Val(1)), backend, [1.0, 2.0]) ≈
          [2.0, 0.0]

    # Matrix functions
    some_matrix = [0.7 0.1; 0.4 0.2]
    f(x) = sum(exp.(x), dims = 1)
    dfdx1(x) = derivative(f, x, [1.0 1.0; 0.0 0.0], Val(1))
    dfdx2(x) = derivative(f, x, [0.0 0.0; 1.0 1.0], Val(1))
    res(x) = sum(dfdx1(x) .+ 2 * dfdx2(x))
    grad = DI.gradient(res, backend, some_matrix)
    @test grad ≈ [1 0; 0 2] * exp.(some_matrix)
end

@testset "Zygote-over-TaylorDiff on different variable" begin
    linear_model(x, p, b) = exp.(b + p * x + b)[1]
    loss_taylor(x, p, b, v) = derivative(x -> linear_model(x, p, b), x, v, Val(1))
    ε = cbrt(eps(Float64))
    loss_finite(x, p, b, v) = (linear_model(x + ε * v, p, b) -
                               linear_model(x - ε * v, p, b)) / (2 * ε)
    let some_x = [0.58, 0.36], some_v = [0.23, 0.11], some_p = [0.49 0.96], some_b = [0.88]
        @test DI.gradient(
            p -> loss_taylor(some_x, p, some_b, some_v), backend, some_p) ≈
              DI.gradient(
            p -> loss_finite(some_x, p, some_b, some_v), backend, some_p)
    end
end
