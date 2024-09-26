using LinearAlgebra
import Zygote # use qualified import to avoid conflict with TaylorDiff

@testset "Zygote-over-TaylorDiff on same variable" begin
    # Scalar functions
    some_number = 0.7
    some_numbers = [0.3 0.4 0.1;]
    for f in (exp, log, sqrt, sin, asin, sinh, asinh, x -> x^3)
        @test Zygote.gradient(derivative, f, some_number, 2)[2] ≈
              derivative(f, some_number, 3)
        @test Zygote.jacobian(broadcast, derivative, f, some_numbers, 2)[3] ≈
              diagm(vec(derivative.(f, some_numbers, 3)))
    end

    # Vector functions
    g(x) = x[1] * x[1] + x[2] * x[2]
    @test Zygote.gradient(derivative, g, [1.0, 2.0], [1.0, 0.0], 1)[2] ≈ [2.0, 0.0]

    # Matrix functions
    some_matrix = [0.7 0.1; 0.4 0.2]
    f(x) = sum(exp.(x), dims = 1)
    dfdx1(x) = derivative(f, x, [1.0, 0.0], 1)
    dfdx2(x) = derivative(f, x, [0.0, 1.0], 1)
    res(x) = sum(dfdx1(x) .+ 2 * dfdx2(x))
    grads = Zygote.gradient(res, some_matrix)
    @test grads[1] ≈ [1 0; 0 2] * exp.(some_matrix)
end

@testset "Zygote-over-TaylorDiff on different variable" begin
    Zygote.gradient(
        p -> derivative(x -> sum(exp.(x + p)), [1.0, 1.0], [1.0, 0.0], 1), [0.5, 0.7])
    Zygote.gradient(
        p -> derivative(x -> sum(exp.(p + x)), [1.0, 1.0], [1.0, 0.0], 1), [0.5, 0.7])
    linear_model(x, p, b) = exp.(b + p * x + b)[1]
    some_x, some_v, some_p, some_b = [0.58, 0.36], [0.23, 0.11], [0.49 0.96], [0.88]
    loss_taylor(p) = derivative(x -> linear_model(x, p, some_b), some_x, some_v, 1)
    ε = cbrt(eps(Float64))
    loss_finite(p) =
        let f = x -> linear_model(x, p, some_b)
            (f(some_x + ε * some_v) - f(some_x - ε * some_v)) / 2ε
        end
    @test Zygote.gradient(loss_taylor, some_p)[1] ≈ Zygote.gradient(loss_finite, some_p)[1]
end
