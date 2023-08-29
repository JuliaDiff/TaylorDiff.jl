
@testset "Scalar" begin
    g(x) = x^3
    @test derivative(g, 1.0, 1) ≈ 3
    @test derivative(g, [2.0 3.0], 1) ≈ [12.0 27.0]
end
