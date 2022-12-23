using FiniteDifferences

@testset "No derivative or linear" begin
    some_number = 3.7
    for f in (+, -, zero, one, adjoint, conj, deg2rad, rad2deg), order in (2, 4)
        @test derivative(f, some_number, order) ≈ 0.0
    end
end

@testset "Unary functions" begin
    some_number = 3.7
    for f in (exp, expm1, exp2, exp10, sin, cos, sqrt, cbrt, inv), order in (2, 4)
        fdm = central_fdm(12, order)
        @test derivative(f, some_number, order)≈fdm(f, some_number) rtol=1e-6
    end
end

@testset "Codegen" begin
    some_number = 0.6
    for f in (log, ), order in (2, 4)
        fdm = central_fdm(12, order, max_range=0.5)
        @test derivative(f, some_number, order)≈fdm(f, some_number) rtol=1e-6
    end
end

@testset "Binary functions" begin
    some_number, another_number = 1.9, 2.6
    for f in (*, /), order in (2, 4)
        fdm = central_fdm(12, order)
        closure = x -> exp(f(x, another_number))
        @test derivative(closure, some_number, order)≈fdm(closure, some_number) rtol=1e-6
    end
    for f in (x -> x^7, x -> x^another_number), order in (2, 4)
        fdm = central_fdm(12, order)
        @test derivative(f, some_number, order)≈fdm(f, some_number) rtol=1e-6
    end
end
