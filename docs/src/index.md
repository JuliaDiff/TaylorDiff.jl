```@meta
CurrentModule = TaylorDiff
```

# TaylorDiff.jl

[TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl) is an automatic differentiation (AD) package for efficient and composable higher-order derivatives, implemented with operator-overloading on Taylor polynomials.

Disclaimer: this project is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!

## Features

TaylorDiff.jl is designed with the following goals in head:

- Linear scaling with the order of differentiation (while naively composing first-order differentiation would result in exponential scaling)
- Same performance with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) on first order and second order, so there is no penalty in drop-in replacement
- Capable for calculating exact derivatives in physical models with ODEs and PDEs
- Composable with other AD systems like [Zygote.jl](https://github.com/FluxML/Zygote.jl), so that the above models evaluated with TaylorDiff can be further optimized with gradient-based optimization techniques

TaylorDiff.jl is fast! See our dedicated [benchmarks](https://benchmark.tansongchen.com/TaylorDiff.jl) page for comparison with other packages in various tasks.

## Installation

```bash
] add TaylorDiff
```

## Usage

```julia
using TaylorDiff

x = 0.1
derivative(sin, x, Val(10)) # scalar derivative
v, direction = [3.0, 4.0], [1.0, 0.0]
derivative(x -> sum(exp.(x)), v, direction, Val(2)) # directional derivative
```

Please see our [documentation](https://juliadiff.org/TaylorDiff.jl) for more details.

## Related Projects

- [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl): a systematic treatment of Taylor polynomials in one and several variables, but its mutating and scalar code isn't great for speed and composability with other packages
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl): well-established and robust operator-overloading based forward-mode AD, where higher-order derivatives can be achieved by nesting first-order derivatives
- [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl): next-generation source-code transformation based forward-mode and reverse-mode AD, designed with support for higher-order derivatives in mind; but the higher-order functionality is currently only a proof-of-concept
- [`jax.jet`](https://jax.readthedocs.io/en/latest/jax.experimental.jet.html): an experimental (and unmaintained) implementation of Taylor-mode automatic differentiation in JAX, sharing the same underlying algorithm with this project

## Citation

```bibtex
@software{tan2022taylordiff,
  author = {Tan, Songchen},
  title = {TaylorDiff.jl: Fast Higher-order Automatic Differentiation in Julia},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JuliaDiff/TaylorDiff.jl}}
}
```
