using Zygote

@testset "Zygote compatibility" begin @test gradient(x -> derivative(x -> x * x, x, 1),
                                                     5.0)[1] ≈ 2.0
    # @test gradient(x -> derivative(g, x, 1, 1), [1., 2.])[1] ≈ [2., 0.]
end
