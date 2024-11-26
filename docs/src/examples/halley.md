# Efficient Halley's method for nonlinear solving

## Introduction

Say we have a system of $n$ equations with $n$ unknowns $f(x)=0$, and $f\in \mathbb R^n\to\mathbb R^n$ is sufficiently smooth.

Given a initial guess $x_0$, Newton's method finds a solution by iterating like

$$x_{i+1}=x_i-J(x_i)^{-1}f(x_i)$$

and this method converges quadratically.

We can make it converge faster using higher-order derivative information. For example, Halley's method iterates like

$$x_{i+1}=x_i-(a_i\odot a_i)\oslash(a_i-b_i/2)$$

where the vector multiplication and division $\odot,\oslash$ are defined element-wise, and term $a_i$ and $b_i$ are defined by $J(x_i)a_i = f(x_i)$ and $J(x_i)b_i = H(x_i)a_ia_i$.

Halley's method is proved to converge cubically, which is faster than Newton's method. Here, we demonstrate that with TaylorDiff.jl, you can compute the Hessian-vector-vector product $H(x_i)a_ia_i$ very efficiently, such that the Halley's method is almost as cheap as Newton's method per iteration.

## Implementation

We first define the two iteration schemes mentioned above:

```@example 1
using TaylorDiff, LinearAlgebra
import ForwardDiff

function newton(f, x, p; tol = 1e-12, maxiter = 100)
    fp = Base.Fix2(f, p)
    for i in 1:maxiter
        fx = fp(x)
        error = norm(fx)
        println("Iteration $i: x = $x, f(x) = $fx, error = $error")
        error < tol && return
        J = ForwardDiff.jacobian(fp, x)
        a = J \ fx
        @. x -= a
    end
end

function halley(f, x, p; tol = 1e-12, maxiter = 100)
    fp = Base.Fix2(f, p)
    for i in 1:maxiter
        fx = f(x, p)
        error = norm(fx)
        println("Iteration $i: x = $x, f(x) = $fx, error = $error")
        error < tol && return
        J = ForwardDiff.jacobian(fp, x)
        a = J \ fx
        hvvp = derivative(fp, x, a, Val(2))
        b = J \ hvvp
        @. x -= (a * a) / (a - b / 2)
    end
end
```

Note that in Halley's method, the hessian-vector-vector product is computed with `derivative(fp, x, a, Val(2))`. It is guaranteed that asymptotically this is only taking 2x more time compared to evaluating `fp(x)` itself.

Now we define some test function:

```@example 1
f(x, p) = x .* x - p
```

The Newton's method takes 6 iterations to converge:

```@example 1
newton(f, [1., 1.], [2., 2.])
```

While the Halley's method takes 4 iterations to converge:

```@example 1
halley(f, [1., 1.], [2., 2.])
```