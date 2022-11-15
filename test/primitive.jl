

@testset "Primitive functions" begin
    @test derivative(exp, 1., 10) ≈ exp(1.)
end
