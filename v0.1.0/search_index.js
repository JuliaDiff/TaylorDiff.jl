var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"API for TaylorDiff.","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [TaylorDiff]","category":"page"},{"location":"api/#TaylorDiff.TaylorScalar","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T <: Number, N}\n\nRepresentation of Taylor polynomials.\n\nFields\n\nvalue::NTuple{N, T}: i-th element of this stores the (i-1)-th derivative\n\n\n\n\n\n","category":"type"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{N}, Tuple{T}, Tuple{T, T}} where {T<:Number, N}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T, N}(x::T, d::T) where {T <: Number, N}\n\nConstruct a seed with arbitrary first-order perturbation.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.TaylorScalar-Union{Tuple{N}, Tuple{T}} where {T<:Number, N}","page":"API","title":"TaylorDiff.TaylorScalar","text":"TaylorScalar{T, N}(x::T) where {T <: Number, N}\n\nConstruct a seed with unit first-order perturbation.\n\n\n\n\n\n","category":"method"},{"location":"api/#TaylorDiff.derivative","page":"API","title":"TaylorDiff.derivative","text":"derivative(f::Function, x::T, order::Int64)\nderivative(f::F, x::T, ::Val{N})\n\nComputes order-th derivative of f w.r.t. x.\n\nderivative(f::Function, x::Vector{T}, l::Vector{T}, order::Int64)\nderivative(f::F, x::Vector{T}, l::Vector{T}, ::Val{N})\n\nComputes order-th directional derivative of f w.r.t. x in direction l.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TaylorDiff","category":"page"},{"location":"#TaylorDiff.jl","page":"Home","title":"TaylorDiff.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TaylorDiff.jl is an automatic differentiation (AD) library for efficient and composable higher-order derivatives, implemented with forward evaluation of overloaded function on Taylor polynomials. It is designed with the following goals in head:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Linear scaling with the order of differentiation (while naively composing first-order differentiation would result in exponential scaling)\nSame performance with ForwardDiff.jl on first order, so there is no penalty in drop-in replacement\nCapable for calculating exact derivatives in physical models with ODEs and PDEs\nComposable with other AD systems like Zygote.jl, so that the above models evaluated with TaylorDiff can be further optimized with gradient-based optimization techniques","category":"page"},{"location":"","page":"Home","title":"Home","text":"This project is still in early alpha stage, and APIs can change any time in the future. Discussions and potential use cases are extremely welcome!","category":"page"},{"location":"#Related-Projects","page":"Home","title":"Related Projects","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This project start from TaylorSeries.jl and re-implement the Taylor mode automatic differentiation primarily for high-order differentiation in solving ODEs and PDEs.","category":"page"}]
}