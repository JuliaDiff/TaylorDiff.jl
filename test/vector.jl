
@testset "Vector" begin
    g(x) = x[1] * x[1] + x[2] * x[2]    
    @test derivative(g, [1., 2.], [1., 0.], 1) ≈ 2.
end
