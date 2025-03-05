using TaylorDiff

f(x, y) = exp(x) * exp(y)
x0, y0 = 0.0, 0.0
x_dx = TaylorScalar{2}(x0, one(x0))
x_dxdy = TaylorScalar{2}(x_dx, zero(x_dx))
y_dx = TaylorScalar{2}(y0)
y_dxdy = TaylorScalar{2}(y_dx, one(y_dx))

result = f(x_dxdy, y_dxdy)
map(TaylorDiff.flatten, TaylorDiff.flatten(result))
