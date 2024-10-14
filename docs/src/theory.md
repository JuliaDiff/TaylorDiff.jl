```@meta
CurrentModule = TaylorDiff
```

# Theory

TaylorDiff.jl is an operator-overloading based forward-mode automatic differentiation (AD) package. "Forward-mode" implies that the basic capability of this package is that, for function $f:\mathbb R^n\to\mathbb R^m$, place to evaluate derivative $x\in\mathbb R^n$ and direction $l\in\mathbb R^n$, we compute
$$
f(x),\partial f(x)\times v,\partial^2f(x)\times v\times v,\cdots,\partial^pf(x)\times v\times\cdots\times v
$$
i.e., the function value and the directional derivative up to order $p$. This notation might be unfamiliar to Julia users that had experience with other AD packages, but $\partial f(x)$ is simply the jacobian $J$, and $\partial f(x)\times v$ is simply the Jacobian-vector product (jvp). In other words, this is a simple generalization of Jacobian-vector product to Hessian-vector-vector product, and to even higher orders.

The main advantage of doing this instead of doing $p$ first-order Jacobian-vector products is that nesting first-order AD results in expential scaling w.r.t $p$, while this method, also known as Taylor mode, should be (almost) linear scaling w.r.t $p$. We will see the reason of this claim later.

In order to achieve this, assuming that $f$ is a nested function $f_k\circ\cdots\circ f_2\circ f_1$, where each $f_i$ is a basic and simple function, or called "primitives". We need to figure out how to propagate the derivatives through each step. In first order AD, this is achieved by the "dual" pair $x_0+x_1\varepsilon$, where $\varepsilon^2=0$, and for each primitive we make a method overload
$$
f(x_0+x_1\varepsilon)=f(x_0)+\partial f(x_0) x_1\varepsilon
$$
Similarly in higher-order AD, we need for each primitive a method overload for a truncated Taylor polynomial up to order $p$, and in this polynomial we will use $t$ instead of $\varepsilon$ to denote the sensitivity. "Truncated" means $t^{p+1}=0$, similar as what we defined for dual numbers. So
$$
f(x_0+x_1t+x_2t^2+\cdots+x_pt^p)=?
$$
What is the math expression that we should put into the question mark? That specific expression is called the "pushforward rule", and we will talk about how to derive the pushforward rule below.

## Arithmetic of polynomials

Before deriving pushforward rules, let's first introduce several basic properties of polynomials.

If $x(t)$ and $y(t)$ are both truncated Taylor polynomials, i.e.
$$
\begin{aligned}
x&=x_0+x_1t+\cdots+x_pt^p\\
y&=y_0+y_1t+\cdots+y_pt^p
\end{aligned}
$$
Then it's obvious that the polynomial addition and subtraction should be
$$
(x\pm y)_k=x_k\pm y_k
$$
And with some derivation we can also get the polynomial multiplication rule
$$
(x\times y)_k=\sum_{i=0}^kx_iy_{k-i}
$$
The polynomial division rule is less obvious, but if $x/y=z$, then equivalently $x=yz$, i.e.
$$
\left(\sum_{i=0}^py_it^i\right)\left(\sum_{i=0}^pz_it^i\right)=\sum_{i=0}^px_it^i
$$
if we relate the coefficient of $t^k$ on both sides we get
$$
\sum_{i=0}^k z_iy_{k-i}=x_k
$$
so, equivalently,
$$
z_k=\frac1{y_0}\left(x_k-\sum_{i=0}^{k-1}z_iy_{k-1}\right)
$$
This is a recurrence relation, which means that we can first get $z_0=x_0/y_0$, and then get $z_1$ using $z_0$, and then get $z_2$ using $z_0,z_1$ etc.

## Pushforward rule for elementary functions

Let's now consider how to derive the pushforward rule for elementary functions. We will use $\exp$ and $\log$ as two examples.

If $x(t)$ is a polynomial and we want to get $e(t)=\exp(x(t))$, we can actually get that by formulating an ordinary differential equation:
$$
e'(t)=\exp(x(t))x'(t);\quad  e_0=\exp(x_0)
$$
If we expand both $e$ and $x$ in the equation, we will get
$$
\sum_{i=1}^pie_it^{i-1}=\left(\sum_{i=0}^{p-1} e_it^i\right)\left(\sum_{i=1}^pix_it^{i-1}\right)
$$
relating the coefficient of $t^{k-1}$ on both sides, we get
$$
ke_k=\sum_{i=0}^{k-1}e_i\times (k-i)x_{k-i}
$$
This is, again, a recurrence relation, so we can get $e_1,\cdots,e_p$ step-by-step.

If $x(t)$ is a polynomial and we want to get $l(t)=\log(x(t))$, we can actually get that by formulating an ordinary differential equation:
$$
l'(t)=\frac1xx'(t);\quad  l_0=\log(x_0)
$$
If we expand both $l$ and $x$ in the equation, the RHS is simply polynomial divisions, and we get
$$
l_k=\frac1{x_0}\left(x_k-\frac1k\sum_{i=1}^{k-1}il_ix_{k-j}\right)
$$

---

Now notice the difference between the rule for $\exp$ and $\log$: the derivative of exponentiation is itself, so we can obtain from recurrence relation; the derivative of logarithm is $1/x$, an algebraic expression in $x$, so it can be directly computed. Similarly, we have $(\tan x)'=1+\tan^2x$ but $(\arctan x)'=(1+x^2)^{-1}$. We summarize (omitting proof) that

- Every $\exp$-like function (like $\sin$, $\cos$, $\tan$, $\sinh$, ...)'s derivative is somehow recursive
- Every $\log$-like function (like $\arcsin$,  $\arccos$, $\arctan$, $\operatorname{arcsinh}$, ...)'s derivative is algebraic

So all of the elementary functions have an easy pushforward rule that can be computed within $O(p^2)$ time. Note that this is an elegant and straightforward corollary from the definition of "elementary function" in differential algebra.

## Generic pushforward rule

For a generic $f(x)$, if we don't bother deriving the specific recurrence rule for it, we can still automatically generate pushforward rule in the following manner. Let's denote the derivative of $f$ w.r.t $x$ to be $d(x)$, then for $f(t)=f(x(t))$  we have
$$
f'(t)=d(x(t))x'(t);\quad f(0)=f(x_0)
$$
when we expand $f$ and $x$ up to order $p$ into this equation, we notice that only order $p-1$ is needed for $d(x(t))$. In other words, we turn a problem of finding $p$-th order pushforward for $f$, to a problem of finding $p-1$-th order pushforward for $d$, and we can recurse down to the first order. The first-order derivative expressions are captured from ChainRules.jl, which made this process fully automatic.

This strategy is in principle equivalent to nesting first-order differentiation, which could potentially leads to exponential scaling; however, in practice there is a huge difference. This generation of pushforward rule happens at **compile time**, which gives the compiler a chance to check redundant expressions and optimize it down to quadratic time. Compiler has stack limits but this should work for at least up to order 100.

In the current implementation of TaylorDiff.jl, all $\log$-like functions' pushforward rules are generated by this strategy, since their derivatives are simple algebraic expressions; some $\exp$-like functions, like sinh, is also generated; the most-often-used several $\exp$-like functions are hand-written with hand-derived recurrence relations.

If you find that the code generated by this strategy is slow, please file an issue and we will look into it.
