using Zygote

@testset "Zygote compatibility" begin @test gradient(x -> derivative(x -> x^4, x, 2), 5.0)[1] ≈
                                            120.0
    # @test gradient(x -> derivative(g, x, 1, 1), [1., 2.])[1] ≈ [2., 0.]
end
