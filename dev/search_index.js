var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"API for TaylorDiff.","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [TaylorDiff]","category":"page"},{"location":"api/#TaylorDiff.TaylorArray","page":"API","title":"TaylorDiff.TaylorArray","text":"TaylorArray{T, N, A, P}\n\nRepresentation of Taylor polynomials in array mode.\n\nFields\n\nvalue::A: zeroth order coefficient\npartials::NTuple{P, A}: i-th element of this stores the i-th derivative\n\n\n\n\n\n","category":"type"},{"location":"api/#TaylorDiff.TaylorScalar","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T, P}\n\nRepresentation of Taylor polynomials.\n\nFields\n\nvalue::T: zeroth order coefficient\npartials::NTuple{P, T}: i-th element of this stores the i-th derivative\n\n\n\n\n\n","category":"type"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{P}, Tuple{T}, Tuple{T, T}} where {T, P}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{P}(value::T, seed::T)\n\nConvenience function: construct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{P}, Tuple{T}} where {T, P}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{P}(value::T) where {T, P}\n\nConvenience function: construct a Taylor polynomial with zeroth order coefficient.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.can_taylorize-Tuple{Type{<:Real}}","page":"API","title":"TaylorDiff.can_taylorize","text":"TaylorDiff.can_taylorize(V::Type)\n\nDetermines whether the type V is allowed as the scalar type in a Dual. By default, only <:Real types are allowed.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.derivative","page":"API","title":"TaylorDiff.derivative","text":"derivative(f, x, ::Val{P})\nderivative(f, x, l, ::Val{P})\nderivative(f!, y, x, l, ::Val{P})\n\nComputes P-th directional derivative of f w.r.t. vector x in direction l. If x is a Number, the direction l can be omitted.\n\n\n\n\n\n","category":"function"},{"location":"api/#TaylorDiff.derivative!","page":"API","title":"TaylorDiff.derivative!","text":"derivative!(result, f, x, l, ::Val{P})\nderivative!(result, f!, y, x, l, ::Val{P})\n\nIn-place derivative calculation APIs. result is expected to be pre-allocated and have the same shape as y.\n\n\n\n\n\n","category":"function"},{"location":"api/#TaylorDiff.derivatives","page":"API","title":"TaylorDiff.derivatives","text":"derivatives(f, x, l, ::Val{P})\nderivatives(f!, y, x, l, ::Val{P})\n\nComputes all derivatives of f at x up to order P.\n\n\n\n\n\n","category":"function"},{"location":"api/#TaylorDiff.get_term_raiser-Tuple{Any}","page":"API","title":"TaylorDiff.get_term_raiser","text":"Pick a strategy for raising the derivative of a function. If the derivative is like 1 over something, raise with the division rule; otherwise, raise with the multiplication rule.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.@immutable-Tuple{Any}","page":"API","title":"TaylorDiff.@immutable","text":"immutable(def)\n\nTransform a function definition to a @generated function.\n\nAllocations are removed by replacing the output with scalar variables;\nLoops are unrolled;\nIndices are modified to use 1-based indexing;\n\n\n\n\n\n","category":"macro"},{"location":"theory/","page":"Theory","title":"Theory","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"theory/#Theory","page":"Theory","title":"Theory","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"TaylorDiff.jl is an operator-overloading based forward-mode automatic differentiation (AD) package. \"Forward-mode\" implies that the basic capability of this package is that, for function fmathbb R^ntomathbb R^m, place to evaluate derivative xinmathbb R^n and direction linmathbb R^n, we compute","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(x)partial f(x)times vpartial^2f(x)times vtimes vcdotspartial^pf(x)times vtimescdotstimes v","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"i.e., the function value and the directional derivative up to order p.  This notation might be unfamiliar to Julia users that had experience with other AD packages, but partial f(x) is simply the jacobian J, and partial f(x)times v is simply the Jacobian-vector product (JVP). In other words, this is a simple generalization of Jacobian-vector product to Hessian-vector-vector product, and to even higher orders.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The main advantage of doing this instead of doing p first-order Jacobian-vector products is that nesting first-order AD results in exponential scaling w.r.t p, while this method, also known as Taylor mode, should scale (almost) linearly w.r.t p.  We will see the reason of this claim later.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"In order to achieve this, we assume that f is a nested function f_kcirccdotscirc f_2circ f_1, where each f_i is a basic and simple function, also called \"primitive\".  We need to figure out how to propagate the derivatives through each step.  In first order AD, this is achieved by the \"dual\" pair x_0+x_1varepsilon, where varepsilon^2=0, and for each primitive we make a method overload","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(x_0+x_1varepsilon)=f(x_0)+partial f(x_0) x_1varepsilon","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Similarly in higher-order AD, we need for each primitive a method overload for a truncated Taylor polynomial up to order p, and in this polynomial we will use t instead of varepsilon to denote the sensitivity. \"Truncated\" means t^p+1=0, similar as what we defined for dual numbers. So","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(x_0+x_1t+x_2t^2+cdots+x_pt^p)=","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"What is the math expression that we should put into the question mark?  That specific expression is called the \"pushforward rule\", and we will talk about how to derive the pushforward rule below.","category":"page"},{"location":"theory/#Arithmetic-of-polynomials","page":"Theory","title":"Arithmetic of polynomials","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"Before deriving pushforward rules, let's first introduce several basic properties of polynomials.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If x(t) and y(t) are both truncated Taylor polynomials, i.e.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"beginaligned\nx=x_0+x_1t+cdots+x_pt^p\ny=y_0+y_1t+cdots+y_pt^p\nendaligned","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Then it's obvious that the polynomial addition and subtraction should be","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"(xpm y)_k=x_kpm y_k","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"And with some derivation we can also get the polynomial multiplication rule","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"(xtimes y)_k=sum_i=0^kx_iy_k-i","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"The polynomial division rule is less obvious, but if xy=z, then equivalently x=yz, i.e.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"left(sum_i=0^py_it^iright)left(sum_i=0^pz_it^iright)=sum_i=0^px_it^i","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"if we relate the coefficient of t^k on both sides we get","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"sum_i=0^k z_iy_k-i=x_k","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"so, equivalently,","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"z_k=frac1y_0left(x_k-sum_i=0^k-1z_iy_k-1right)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"This is a recurrence relation, which means that we can first get z_0=x_0y_0, and then get z_1 using z_0, and then get z_2 using z_0z_1 etc.","category":"page"},{"location":"theory/#Pushforward-rule-for-elementary-functions","page":"Theory","title":"Pushforward rule for elementary functions","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"Let's now consider how to derive the pushforward rule for elementary functions. We will use exp and log as two examples.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If x(t) is a polynomial and we want to get e(t)=exp(x(t)), we can actually get that by formulating an ordinary differential equation:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"e(t)=exp(x(t))x(t)quad  e_0=exp(x_0)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If we expand both e and x in the equation, we will get","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"sum_i=1^pie_it^i-1=left(sum_i=0^p-1 e_it^iright)left(sum_i=1^pix_it^i-1right)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"relating the coefficient of t^k-1 on both sides, we get","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"ke_k=sum_i=0^k-1e_itimes (k-i)x_k-i","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"This is, again, a recurrence relation, so we can get e_1cdotse_p step-by-step.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If x(t) is a polynomial and we want to get l(t)=log(x(t)), we can actually get that by formulating an ordinary differential equation:","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"l(t)=frac1xx(t)quad  l_0=log(x_0)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If we expand both l and x in the equation, the RHS is simply polynomial divisions, and we get","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"l_k=frac1x_0left(x_k-frac1ksum_i=1^k-1il_ix_k-jright)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Now notice the difference between the rule for exp and log: the derivative of exponentiation is itself, so we can obtain from recurrence relation; the derivative of logarithm is 1x, an algebraic expression in x, so it can be directly computed.  Similarly, we have (tan x)=1+tan^2x but (arctan x)=(1+x^2)^-1. We summarize (omitting proof) that","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"Every exp-like function (like sin, cos, tan, sinh, ...)'s derivative is somehow recursive\nEvery log-like function (like arcsin,  arccos, arctan, operatornamearcsinh, ...)'s derivative is algebraic","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"So all of the elementary functions have an easy pushforward rule that can be computed within O(p^2) time. Note that this is an elegant and straightforward corollary from the definition of \"elementary function\" in differential algebra.","category":"page"},{"location":"theory/#Generic-pushforward-rule","page":"Theory","title":"Generic pushforward rule","text":"","category":"section"},{"location":"theory/","page":"Theory","title":"Theory","text":"For a generic f(x), if we don't bother deriving the specific recurrence rule for it, we can still automatically generate a pushforward rule in the following manner. Let's denote the derivative of f w.r.t x to be d(x), then for f(t)=f(x(t))  we have","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"f(t)=d(x(t))x(t)quad f(0)=f(x_0)","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"when we expand f and x up to order p into this equation, we notice that only order p-1 is needed for d(x(t)).  In other words, we turn a problem of finding p-th order pushforward for f, to a problem of finding (p-1)-th order pushforward for d, and we can recurse down to the first order.  The first-order derivative expressions are captured from ChainRules.jl, which makes this process fully automatic.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"This strategy is in principle equivalent to nesting first-order differentiation, which could potentially lead to exponential scaling; however, in practice there is a huge difference.  This generation of pushforward rules happens at compile time, which gives the compiler a chance to check redundant expressions and optimize it down to quadratic time.  The compiler has stack limits but this should work at least up to order 100.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"In the current implementation of TaylorDiff.jl, all log-like functions' pushforward rules are generated by this strategy, since their derivatives are simple algebraic expressions; some exp-like functions, like sinh, are also generated; several of the most-often-used exp-like functions are hand-written with hand-derived recurrence relations.","category":"page"},{"location":"theory/","page":"Theory","title":"Theory","text":"If you find that the code generated by this strategy is slow, please file an issue and we will look into it.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"#TaylorDiff.jl","page":"Home","title":"TaylorDiff.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is an automatic differentiation (AD) package for efficient and composable higher-order derivatives, implemented with operator-overloading on Taylor polynomials.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Disclaimer: this project is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is designed with the following goals in head:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Linear scaling with the order of differentiation (while naively composing first-order differentiation would result in exponential scaling)\nSame performance with ForwardDiff.jl on first order and second order, so there is no penalty in drop-in replacement\nCapable for calculating exact derivatives in physical models with ODEs and PDEs\nComposable with other AD systems like Zygote.jl, so that the above models evaluated with TaylorDiff can be further optimized with gradient-based optimization techniques","category":"page"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is fast! See our dedicated benchmarks page for comparison with other packages in various tasks.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"] add TaylorDiff","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using TaylorDiff\n\nx = 0.1\nderivative(sin, x, Val(10)) # scalar derivative\nv, direction = [3.0, 4.0], [1.0, 0.0]\nderivative(x -> sum(exp.(x)), v, direction, Val(2)) # directional derivative","category":"page"},{"location":"","page":"Home","title":"Home","text":"Please see our documentation for more details.","category":"page"},{"location":"#Related-Projects","page":"Home","title":"Related Projects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorSeries.jl: a systematic treatment of Taylor polynomials in one and several variables, but its mutating and scalar code isn't great for speed and composability with other packages\nForwardDiff.jl: well-established and robust operator-overloading based forward-mode AD, where higher-order derivatives can be achieved by nesting first-order derivatives\nDiffractor.jl: next-generation source-code transformation based forward-mode and reverse-mode AD, designed with support for higher-order derivatives in mind; but the higher-order functionality is currently only a proof-of-concept\njax.jet: an experimental (and unmaintained) implementation of Taylor-mode automatic differentiation in JAX, sharing the same underlying algorithm with this project","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"@software{tan2022taylordiff,\n  author = {Tan, Songchen},\n  title = {TaylorDiff.jl: Fast Higher-order Automatic Differentiation in Julia},\n  year = {2022},\n  publisher = {GitHub},\n  journal = {GitHub repository},\n  howpublished = {\\url{https://github.com/JuliaDiff/TaylorDiff.jl}}\n}","category":"page"},{"location":"examples/halley/#Efficient-Halley's-method-for-nonlinear-solving","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"","category":"section"},{"location":"examples/halley/#Introduction","page":"Efficient Halley's method for nonlinear solving","title":"Introduction","text":"","category":"section"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"Say we have a system of n equations with n unknowns f(x)=0, and fin mathbb R^ntomathbb R^n is sufficiently smooth.","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"Given a initial guess x_0, Newton's method finds a solution by iterating like","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"x_i+1=x_i-J(x_i)^-1f(x_i)","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"and this method converges quadratically.","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"We can make it converge faster using higher-order derivative information. For example, Halley's method iterates like","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"x_i+1=x_i-(a_iodot a_i)oslash(a_i-b_i2)","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"where the vector multiplication and division odotoslash are defined element-wise, and term a_i and b_i are defined by J(x_i)a_i = f(x_i) and J(x_i)b_i = H(x_i)a_ia_i.","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"Halley's method is proved to converge cubically, which is faster than Newton's method. Here, we demonstrate that with TaylorDiff.jl, you can compute the Hessian-vector-vector product H(x_i)a_ia_i very efficiently, such that the Halley's method is almost as cheap as Newton's method per iteration.","category":"page"},{"location":"examples/halley/#Implementation","page":"Efficient Halley's method for nonlinear solving","title":"Implementation","text":"","category":"section"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"We first define the two iteration schemes mentioned above:","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"using TaylorDiff, LinearAlgebra\nimport ForwardDiff\n\nfunction newton(f, x, p; tol = 1e-12, maxiter = 100)\n    fp = Base.Fix2(f, p)\n    for i in 1:maxiter\n        fx = fp(x)\n        error = norm(fx)\n        println(\"Iteration $i: x = $x, f(x) = $fx, error = $error\")\n        error < tol && return\n        J = ForwardDiff.jacobian(fp, x)\n        a = J \\ fx\n        @. x -= a\n    end\nend\n\nfunction halley(f, x, p; tol = 1e-12, maxiter = 100)\n    fp = Base.Fix2(f, p)\n    for i in 1:maxiter\n        fx = f(x, p)\n        error = norm(fx)\n        println(\"Iteration $i: x = $x, f(x) = $fx, error = $error\")\n        error < tol && return\n        J = ForwardDiff.jacobian(fp, x)\n        a = J \\ fx\n        hvvp = derivative(fp, x, a, Val(2))\n        b = J \\ hvvp\n        @. x -= (a * a) / (a - b / 2)\n    end\nend","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"Note that in Halley's method, the hessian-vector-vector product is computed with derivative(fp, x, a, Val(2)). It is guaranteed that asymptotically this is only taking 2x more time compared to evaluating fp(x) itself.","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"Now we define some test function:","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"f(x, p) = x .* x - p","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"The Newton's method takes 6 iterations to converge:","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"newton(f, [1., 1.], [2., 2.])","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"While the Halley's method takes 4 iterations to converge:","category":"page"},{"location":"examples/halley/","page":"Efficient Halley's method for nonlinear solving","title":"Efficient Halley's method for nonlinear solving","text":"halley(f, [1., 1.], [2., 2.])","category":"page"}]
}
