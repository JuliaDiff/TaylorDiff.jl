
@testset "O-function, O-derivative" begin
    g(x) = x^3
    @test derivative(g, 1.0, Val(1)) ≈ 3

    h(x) = x .^ 3
    @test derivative(h, [2.0 3.0], [1.0 1.0], Val(1)) ≈ [12.0 27.0]

    g1(x) = x[1] * x[1] + x[2] * x[2]
    @test derivative(g1, [1.0, 2.0], [1.0, 0.0], Val(1)) ≈ 2.0

    h1(x) = sum(x, dims = 1)
    @test derivative(h1, [1.0 2.0; 2.0 3.0], [1.0 1.0; 1.0 1.0], Val(1)) ≈ [2.0 2.0]
end

@testset "I-function, O-derivative" begin
    g!(y, x) = begin
        y[1] = x * x
        y[2] = x + 1
    end
    x = 2.0
    y = [0.0, 0.0]
    @test derivative(g!, y, x, 1.0, Val(1)) ≈ [4.0, 1.0]
end

@testset "O-function, I-derivative" begin
    g(x) = x .^ 2
    @test derivative!(zeros(2), g, [1.0, 2.0], [1.0, 0.0], Val(1)) ≈ [2.0, 0.0]
    gzero(x) = [1.0, 1.0]
    @test derivative(gzero, [1.0, 2.0], [1.0, 0.0], Val(1)) == [0.0, 0.0]
end

@testset "I-function, I-derivative" begin
    g!(y, x) = begin
        y[1] = x[1] * x[1]
        y[2] = x[2] * x[2]
    end
    x = [2.0, 3.0]
    y = [0.0, 0.0]
    @test derivative!(y, g!, zeros(2), x, [1.0, 0.0], Val(1)) ≈ [4.0, 0.0]
end
