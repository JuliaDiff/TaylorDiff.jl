
@testset "Vector" begin
    cube(x) = x^3
    @test derivative(cube, 1., 2) ≈ 6
end
