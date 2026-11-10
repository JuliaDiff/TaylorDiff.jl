using FiniteDifferences
using SpecialFunctions
using TaylorDiff.ChainRulesCore: frule, NoTangent, AbstractZero

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

@testset "Higher-order terms match the first-order rules" begin
    # `get_term_raiser` takes the derivative expressions from DiffRules; check them against
    # ChainRules' rules, which are used for the first order
    domain = Dict(
        asec => (1.3, -2.0), acsc => (1.3, -2.0), acosh => (1.3, 2.0), acoth => (1.3, -2.0),
        log => (0.3, 2.0), log10 => (0.3, 2.0), log2 => (0.3, 2.0), asech => (0.3, 0.7),
        erfcinv => (0.3, 1.5))
    for f in (deg2rad, rad2deg, sinh, cosh, tanh, asin, acos, atan, asec, acsc, acot,
            log, log10, log1p, log2, asinh, acosh, atanh, asech, acsch, acoth, abs, sign,
            erf, erfc, erfcinv, erfcx, erfi),
        x in get(domain, f, (0.3, -0.7))

        term, raiser = TaylorDiff.get_term_raiser(f)
        d = @eval let z = $x
            $term
        end
        expected = frule((NoTangent(), 1.0), f, x)[2]
        expected isa AbstractZero && (expected = 0.0)  # e.g., `sign`
        @test (raiser === TaylorDiff.raiseinv ? inv(d) : d) ≈ expected
    end
end

@testset "Derivatives of the form 1 / g are raised by division" begin
    # Guards against changes in the form of the DiffRules expressions that would silently
    # fall back to the slower and less accurate multiplication rule
    for f in (asin, acos, atan, asec, acsc, acot, log, log10, log1p, log2,
        asinh, acosh, atanh, asech, acsch, acoth)
        @test TaylorDiff.get_term_raiser(f)[2] === TaylorDiff.raiseinv
    end
    for f in (deg2rad, rad2deg, sinh, cosh, tanh, abs, sign,
        erf, erfc, erfcinv, erfcx, erfi)
        @test TaylorDiff.get_term_raiser(f)[2] === TaylorDiff.raise
    end
end

@testset "Higher-order derivatives preserve Float32" begin
    for f in (deg2rad, rad2deg, sinh, cosh, tanh, atan, acot, log, log10, log1p, log2,
            atanh, abs, sign, erf, erfc, erfcx, erfi)
        @test derivative(f, 0.3f0, Val(3)) isa Float32
    end
end
