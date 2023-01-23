var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"API for TaylorDiff.","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [TaylorDiff]","category":"page"},{"location":"api/#TaylorDiff.TaylorScalar","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T <: Number, N}\n\nRepresentation of Taylor polynomials.\n\nFields\n\nvalue::NTuple{N, T}: i-th element of this stores the (i-1)-th derivative\n\n\n\n\n\n","category":"type"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{N}, Tuple{T}, Tuple{S}, Tuple{S, S}} where {S<:Number, T<:Number, N}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T, N}(x::S, d::S) where {S <: Number, T <: Number, N}\n\nConstruct a Taylor polynomial with zeroth and first order coefficient, acting as a seed.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{N}, Tuple{T}, Tuple{S}} where {S<:Number, T<:Number, N}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T, N}(x::S) where {S <: Number, T <: Number, N}\n\nConstruct a Taylor polynomial with zeroth order coefficient.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.derivative","page":"API","title":"TaylorDiff.derivative","text":"derivative(f, x::T, order::Int64)\nderivative(f, x::T, ::Val{N})\n\nComputes order-th derivative of f w.r.t. x.\n\nderivative(f, x::Vector{T}, l::Vector{T}, order::Int64)\nderivative(f, x::Vector{T}, l::Vector{T}, ::Val{N})\n\nComputes order-th directional derivative of f w.r.t. x in direction l.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"#TaylorDiff.jl","page":"Home","title":"TaylorDiff.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is an automatic differentiation (AD) package for efficient and composable higher-order derivatives, implemented with operator-overloading on Taylor polynomials.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Disclaimer: this project is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is designed with the following goals in head:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Linear scaling with the order of differentiation (while naively composing first-order differentiation would result in exponential scaling)\nSame performance with ForwardDiff.jl on first order and second order, so there is no penalty in drop-in replacement\nCapable for calculating exact derivatives in physical models with ODEs and PDEs\nComposable with other AD systems like Zygote.jl, so that the above models evaluated with TaylorDiff can be further optimized with gradient-based optimization techniques","category":"page"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is fast! See our dedicated benchmarks page for comparison with other packages in various tasks.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"] add TaylorDiff","category":"page"},{"location":"#Usage","page":"Home","title":"Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using TaylorDiff\n\nx = 0.1\nderivative(sin, x, 10) # scalar derivative\nv, direction = [3.0, 4.0], [1.0, 0.0]\nderivative(x -> sum(exp.(x)), v, direction, 2) # directional derivative","category":"page"},{"location":"","page":"Home","title":"Home","text":"Please see our documentation for more details.","category":"page"},{"location":"#Related-Projects","page":"Home","title":"Related Projects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorSeries.jl: a systematic treatment of Taylor polynomials in one and several variables, but its mutating and scalar code isn't great for speed and composability with other packages\nForwardDiff.jl: well-established and robust operator-overloading based forward-mode AD, where higher-order derivatives can be achieved by nesting first-order derivatives\nDiffractor.jl: next-generation source-code transformation based forward-mode and reverse-mode AD, designed with support for higher-order derivatives in mind; but the higher-order functionality is currently only a proof-of-concept\njax.jet: an experimental (and unmaintained) implementation of Taylor-mode automatic differentiation in JAX, sharing the same underlying algorithm with this project","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"@software{tan2022taylordiff,\n  author = {Tan, Songchen},\n  title = {TaylorDiff.jl: Fast Higher-order Automatic Differentiation in Julia},\n  year = {2022},\n  publisher = {GitHub},\n  journal = {GitHub repository},\n  howpublished = {\\url{https://github.com/JuliaDiff/TaylorDiff.jl}}\n}","category":"page"}]
}
