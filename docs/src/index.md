```@meta
CurrentModule = TaylorDiff
```

# TaylorDiff.jl

[TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl) is an automatic differentiation (AD) library for efficient and composable higher-order derivatives, implemented with forward evaluation of overloaded function on Taylor polynomials. It is designed with the following goals in head:

- Linear scaling with the order of differentiation (while naively composing first-order differentiation would result in exponential scaling)
- Same performance with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) on first order, so there is no penalty in drop-in replacement
- Capable for calculating exact derivatives in physical models with ODEs and PDEs
- Composable with other AD systems like [Zygote.jl](https://github.com/FluxML/Zygote.jl), so that the above models evaluated with TaylorDiff can be further optimized with gradient-based optimization techniques

This project is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!

# Related Projects

This project start from [TaylorSeries.jl](https://github.com/JuliaDiff/TaylorSeries.jl) and re-implement the Taylor mode automatic differentiation primarily for high-order differentiation in solving ODEs and PDEs.
