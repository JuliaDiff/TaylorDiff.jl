using FastInterpolations
import ForwardDiff, TaylorDiff
using BenchmarkTools

x = 0.0:0.1:1.0
y = sin.(π * x)
itp = cubic_interp(x, y)

x0 = 0.45

function analytic(itp, x0)
    return itp(x0; deriv = DerivOp(2))
end
@btime analytic($itp, $x0)

function td(itp, x0)
    t = TaylorDiff.TaylorScalar{2}(x0, one(x0))
    v = itp(t)
    return v.partials[2] * 2
end
@btime td($itp, $x0)

@code_typed analytic(itp, x0)
@code_typed td(itp, x0)

z = [sin(xi) * sin(yi) for xi in x, yi in x]
itp2d = cubic_interp((x, x), z)

function analytic2(itp2d, x0)
    return itp2d((x0, x0); deriv = DerivOp(2, 0))
end
@btime analytic2($itp2d, $x0)

@code_native analytic2(itp2d, x0)

function td2(itp2d, x0)
    t = TaylorDiff.TaylorScalar{2}(x0, one(x0))
    v = itp2d((t, x0))
    return v.partials[2] * 2
end
@code_native td2(itp2d, x0)
