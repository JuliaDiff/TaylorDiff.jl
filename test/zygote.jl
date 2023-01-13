using Zygote

@testset "Zygote for mixed derivative" begin
    some_number = 0.7
    for f in (exp, log, sqrt, sin, asin, sinh, asinh)
        @test gradient(x -> derivative(f, x, 2), some_number)[1] ≈ derivative(f, some_number, 3)
    end
    @test gradient(x -> derivative(x -> x * x, x, 1),
                                                     5.0)[1] ≈ 2.0

    g(x) = x[1] * x[1] + x[2] * x[2]
    @test gradient(x -> derivative(g, x, [1., 0.], 1), [1., 2.])[1] ≈ [2., 0.]
end

@testset "Zygote for parameter optimization" begin
    linear_model(x, p) = exp.(p * x)[1]
    some_x, some_v, some_p = [.58, .36], [.23, .11], [.49 .96]
    loss_taylor(p) = derivative(x -> linear_model(x, p), some_x, some_v, 1)
    ε = cbrt(eps(Float64))
    loss_finite(p) = let f = x -> linear_model(x, p)
        (f(some_x + ε * some_v) - f(some_x - ε * some_v)) / 2ε
    end
    @test gradient(loss_taylor, some_p)[1] ≈ gradient(loss_finite, some_p)[1]
end
