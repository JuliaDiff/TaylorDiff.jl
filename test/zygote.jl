using TaylorDiff

f(x) = x * x / 2
g(x) = x + x
h(x) = x[1] * x[2]

using Zygote

Zygote.gradient(df, 1.)
Zygote.gradient(dg, 1.)
Zygote.gradient(dh, 1.)

derivative(h, [1., 2.], 1)
derivative(h, [1., 2.], 1, 1)
gradient(x -> derivative(h, x, 1), [1., 2.])
gradient(x -> derivative(h, x, 1, 1), [1., 2.])

relu(x) = x > 0 ? x : zero(x)
