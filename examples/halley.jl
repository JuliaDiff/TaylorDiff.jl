# The Jacobi- and Hessian-free Halley method for solving nonlinear equations

using TaylorDiff
using LinearAlgebra
using LinearSolve

function newton(f, x0, p; tol=1e-10, maxiter=100)
    x = x0
    for i in 1:maxiter
        fx = f(x, p)
        error = norm(fx)
        println("Iteration $i: x = $x, f(x) = $fx, error = $error")
        if error < tol
            return x
        end
        get_derivative = (v, u, a, b) -> v .= derivative(x -> f(x, p), x, u, 1)
        operator = FunctionOperator(get_derivative, similar(x), similar(x))
        problem = LinearProblem(operator, -fx)
        sol = solve(problem, KrylovJL_GMRES())
        x += sol.u
    end
    return x
end

function halley(f, x0, p; tol=1e-10, maxiter=100)
    x = x0
    for i in 1:maxiter
        fx = f(x, p)
        error = norm(fx)
        println("Iteration $i: x = $x, f(x) = $fx, error = $error")
        if error < tol
            return x
        end
        get_derivative = (v, u, a, b) -> v .= derivative(x -> f(x, p), x, u, 1)
        operator = FunctionOperator(get_derivative, similar(x), similar(x))
        problem = LinearProblem(operator, -fx)
        a = solve(problem, KrylovJL_GMRES()).u
        Haa = derivative(x -> f(x, p), x, a, 2)
        problem2 = LinearProblem(operator, Haa)
        b = solve(problem2, KrylovJL_GMRES()).u
        x += (a .* a) ./ (a .+ b ./ 2)
    end
    return x
end

f(x, p) = x .* x - p
