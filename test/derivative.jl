
@testset "Derivative" begin
    g(x) = x^3
    @test derivative(g, 1.0, 1) ≈ 3

    h(x) = x.^3
    @test derivative(h, [2.0 3.0], 1) ≈ [12.0 27.0]
end

@testset "Directional derivative" begin
    g(x) = x[1] * x[1] + x[2] * x[2]
    @test derivative(g, [1.0, 2.0], [1.0, 0.0], 1) ≈ 2.0

    h(x) = sum(x, dims=1)
    @test derivative(h, [1.0 2.0; 2.0 3.0], [1.0, 1.0], 1) ≈ [2. 2.]
end
