<h1 align=center>TaylorDiff.jl</h1>

<p align=center>
  <a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active – The project has reached a stable, usable state and is being actively developed." /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="https://juliadiff.org/TaylorDiff.jl/stable/"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable" /></a>
  <a href="https://juliadiff.org/TaylorDiff.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev" /></a>
  <br />
  <a href="https://github.com/JuliaDiff/TaylorDiff.jl/actions/workflows/Test.yml?query=branch%3Amain"><img src="https://img.shields.io/github/actions/workflow/status/JuliaDiff/TaylorDiff.jl/Test.yml?branch=main&label=test" alt="Build Status" /></a>
  <a href="https://codecov.io/gh/JuliaDiff/TaylorDiff.jl"><img src="https://img.shields.io/codecov/c/gh/JuliaDiff/TaylorDiff.jl/main?token=5KYP7K71VQ"/></a>
  <a href="https://benchmark.tansongchen.com/TaylorDiff.jl"><img src="https://img.shields.io/buildkite/2c801728055463e7c8baeeb3cc187b964587235a49b3ed39ab/main.svg?label=benchmark" alt="Benchmark Status" /></a>
  <br />
  <a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/contributor's%20guide-ColPrac-blueviolet" alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" /></a>
  <a href="https://github.com/SciML/SciMLStyle"><img src="https://img.shields.io/badge/code%20style-SciML-blueviolet" alt="SciML Code Style" /></a>
  <a href="https://doi.org/10.5281/zenodo.14226430"><img src="https://zenodo.org/badge/563952901.svg" alt="DOI" /></a>
</p>

<p align=center>
  <a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=563952901&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUshttps://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=563952901&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUs"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces" /></a>
</p>

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
