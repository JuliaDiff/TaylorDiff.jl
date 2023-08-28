
@testset "Vector" begin
    g(x) = x[1] * x[1] + x[2] * x[2]
    @test derivative(g, [1.0, 2.0], [1.0, 0.0], 1) ≈ 2.0
    @test derivative(g, [1.0 2.0; 2.0 3.0], [1.0, 1.0], 1) ≈ [6.0 10.0]
end
