using Zygote, LinearAlgebra

@testset "Zygote for mixed derivative" begin
    some_number = 0.7
    some_numbers = [0.3 0.4 0.1;]
    for f in (exp, log, sqrt, sin, asin, sinh, asinh)
        @test gradient(x -> derivative(f, x, 2), some_number)[1] ≈
              derivative(f, some_number, 3)
        derivative_result = vec(derivative(f, some_numbers, 3))
        @test Zygote.jacobian(x -> derivative(f, x, 2), some_numbers)[1] ≈
              diagm(derivative_result)
    end

    some_matrix = [0.7; 0.1;; 0.4; 0.2]
    f = x -> sum(tanh.(x), dims = 1)
    dfdx1(m, x) = derivative(u -> sum(m(u)), x, [1.0, 0.0], 1)
    dfdx2(m, x) = derivative(u -> sum(m(u)), x, [0.0, 1.0], 1)
    res(m, x) = dfdx1(m, x) .+ 2 * dfdx2(m, x)
    grads = Zygote.gradient(some_matrix) do x
        sum(res(f, x))
    end
    expected_grads = x -> -2 * sinh(x) / cosh(x)^3
    @test grads[1] ≈ [1 0; 0 2] * expected_grads.(some_matrix)

    @test gradient(x -> derivative(x -> x * x, x, 1),
        5.0)[1] ≈ 2.0

    g(x) = x[1] * x[1] + x[2] * x[2]
    @test gradient(x -> derivative(g, x, [1.0, 0.0], 1),
        [1.0, 2.0])[1] ≈ [2.0, 0.0]
end

@testset "Zygote for parameter optimization" begin
    gradient(p -> derivative(x -> sum(exp.(x + p)), [1.0, 1.0], [1.0, 0.0], 1), [0.5, 0.7])
    gradient(p -> derivative(x -> sum(exp.(p + x)), [1.0, 1.0], [1.0, 0.0], 1), [0.5, 0.7])
    linear_model(x, p, b) = exp.(b + p * x + b)[1]
    some_x, some_v, some_p, some_b = [0.58, 0.36], [0.23, 0.11], [0.49 0.96], [0.88]
    loss_taylor(p) = derivative(x -> linear_model(x, p, some_b), some_x, some_v, 1)
    ε = cbrt(eps(Float64))
    loss_finite(p) =
        let f = x -> linear_model(x, p, some_b)
            (f(some_x + ε * some_v) - f(some_x - ε * some_v)) / 2ε
        end
    @test gradient(loss_taylor, some_p)[1] ≈ gradient(loss_finite, some_p)[1]
end
