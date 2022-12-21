
@testset "Unary functions" begin
    @test derivative(exp, 1.0, 10) ≈ exp(1.0)
    @test derivative(expm1, 1.0, 10) ≈ exp(1.0)
    @test derivative(log, 10.0, 1) ≈ 0.1
    @test derivative(log, 10.0, 2) ≈ -0.01
    @test derivative(log, 10.0, 3) ≈ 0.002
    @test derivative(log1p, 9.0, 1) ≈ 0.1
    @test derivative(log1p, 9.0, 2) ≈ -0.01
    @test derivative(log1p, 9.0, 3) ≈ 0.002
    @test derivative(sin, 1.0, 1) ≈ cos(1.0)
    @test derivative(sin, 1.0, 2) ≈ -sin(1.0)
    @test derivative(sin, 1.0, 3) ≈ -cos(1.0)
    @test derivative(sin, 1.0, 4) ≈ sin(1.0)
    @test derivative(cos, 1.0, 1) ≈ -sin(1.0)
    @test derivative(cos, 1.0, 2) ≈ -cos(1.0)
    @test derivative(cos, 1.0, 3) ≈ sin(1.0)
    @test derivative(cos, 1.0, 4) ≈ cos(1.0)
    @test derivative(asin, 0.5, 1) ≈ 2 / √3
    @test derivative(acos, 0.5, 1) ≈ -2 / √3
    @test derivative(atan, 1.0, 1) ≈ 0.5
end

@testset "Binary functions" begin @test derivative(x -> x^3, 3.0, 2) ≈ 18.0 end
