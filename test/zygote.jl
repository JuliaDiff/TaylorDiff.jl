using Zygote

f(x) = x * x * x
g(x) = x[1] * x[1] + x[2] * x[2]

@testset "Zygote compatibility" begin
    @test derivative(f, 1., 2) ≈ 6.
    @test derivative(g, [1., 2.], 1, 1) ≈ 2.
    @test gradient(x -> derivative(f, x, 2), 3.)[1] ≈ 6.
    @test gradient(x -> derivative(g, x, 1, 1), [1., 2.])[1] ≈ [2., 0.]
end
