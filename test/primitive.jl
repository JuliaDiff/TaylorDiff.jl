using FiniteDifferences

@testset "No derivative or linear" begin
    some_number, another_number = 1.9, 2.6
    for f in (+, -, zero, one, adjoint, conj, deg2rad, rad2deg, abs, sign),
        order in (Val(2),)

        @test derivative(f, some_number, order) ≈ 0.0
    end
    for f in (+, -, <, <=, >, >=, ==), order in (Val(2),)
        @test derivative(x -> f(x, another_number), some_number, order) ≈ 0.0
        @test derivative(x -> f(another_number, x), some_number, order) ≈ 0.0
        @test derivative(x -> f(x, x), some_number, order) ≈ 0.0
    end
end

@testset "Unary functions" begin
    some_number = 3.7
    for f in (
            x -> exp(x^2), expm1, exp2, exp10, x -> sin(x^2), x -> cos(x^2), sinpi, cospi,
            sqrt, cbrt,
            inv), order in (1, 4)
        fdm = central_fdm(12, order)
        @test derivative(f, some_number, Val(order))≈fdm(f, some_number) rtol=1e-6
    end
end

@testset "Codegen" begin
    some_number = 0.6
    for f in (log, sinh), order in (1, 4)
        fdm = central_fdm(12, order, max_range = 0.5)
        @test derivative(f, some_number, Val(order))≈fdm(f, some_number) rtol=1e-6
    end
end

@testset "Binary functions" begin
    some_number, another_number = 1.9, 5.6
    for f in (*, /), order in (1, 4)
        fdm = central_fdm(12, order)
        closure = x -> exp(f(x, another_number))
        @test derivative(closure, some_number, Val(order))≈fdm(closure, some_number) rtol=1e-6
    end
    for f in (x -> x^7, x -> x^another_number), order in (1, 2, 4)
        fdm = central_fdm(12, order)
        @test derivative(f, some_number, Val(order))≈fdm(f, some_number) rtol=1e-6
    end
    # for f in (x -> x^7, x -> x^another_number), order in (1, 2)
    #     fdm = forward_fdm(12, order)
    #     @test derivative(f, 0., order)≈fdm(f, 0.) atol=1e-6
    # end
end

@testset "Multi-argument functions" begin
    @test derivative(x -> 1 + 1 / x, 1.0, Val(1))≈-1.0 rtol=1e-6
    @test derivative(x -> (x + 1) / x, 1.0, Val(1))≈-1.0 rtol=1e-6
    @test derivative(x -> x / x, 1.0, Val(1))≈0.0 rtol=1e-6
end

@testset "Corner cases" begin
    offenders = (
        TaylorScalar(Inf, (1.0, 0.0, 0.0)),
        TaylorScalar(Inf, (0.0, 0.0, 0.0)),
        TaylorScalar(1.0, (0.0, 0.0, 0.0)),
        TaylorScalar(1.0, (Inf, 0.0, 0.0)),
        TaylorScalar(0.0, (1.0, 0.0, 0.0)),
        TaylorScalar(0.0, (Inf, 0.0, 0.0))
    )
    f_id = (
        :id => x -> x,
        :add0 => x -> x + 0,
        :sub0 => x -> x - 0,
        :mul1 => x -> x * 1,
        :div1 => x -> x / 1,
        :pow1 => x -> x^1
    )
    for (name, f) in f_id, t in offenders
        @test f(t) == t
    end
end
