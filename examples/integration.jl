using TaylorDiff: TaylorDiff, TaylorScalar, TaylorArray, make_seed, value, partials, flatten
using TaylorSeries
using TaylorIntegration: jetcoeffs!
using BenchmarkTools

"""
No magic, just a type-stable way to generate a new object since TaylorScalar is immutable
"""
function setindex(x::TaylorScalar{T, P}, index, d) where {T, P}
    v = flatten(x)
    ntuple(i -> i == index + 1 ? d : v[i], Val(P + 1)) |> TaylorScalar
end

function setindex(x::TaylorArray{T, N, A, P}, index, d) where {T, N, A, P}
    v = flatten(x)
    ntuple(i -> i == index + 1 ? d : v[i], Val(P + 1)) |> TaylorArray
end

"""
Computes the taylor integration of order P

- `f`: ODE function
- `t`: constructed by TaylorScalar{P}(t0, one(t0))
- `x0`: initial value
- `p`: parameters
"""
function my_jetcoeffs(f, t::TaylorScalar{T, P}, x0, p) where {T, P}
    x = x0 isa AbstractArray ? TaylorArray{P}(x0) : TaylorScalar{P}(x0)
    for index in 1:P # computes x.partials[index]
        fx = f(x, p, t)
        d = index == 1 ? value(fx) : partials(fx)[index - 1] / index
        x = setindex(x, index, d)
    end
    x
end

"""
Computes the taylor integration of order P

- `f!`: ODE function, in non-allocating form
- `t`: constructed by TaylorScalar{P}(t0, one(t0))
- `x0`: initial value
- `p`: parameters
"""
function my_jetcoeffs!(f!, t::TaylorScalar{T, P}, x0, p) where {T, P}
    x = x0 isa AbstractArray ? TaylorArray{P}(x0) : TaylorScalar{P}(x0)
    out = similar(x)
    for index in 1:P # computes x.partials[index]
        f!(out, x, p, t)
        d = index == 1 ? value(out) : partials(out)[index - 1] / index
        x = setindex(x, index, d)
    end
    x
end

function scalar_test()
    f(x, p, t) = x * x
    x0 = 0.1
    t0 = 0.0
    P = 6

    # TaylorIntegration test
    t = t0 + Taylor1(typeof(t0), P)
    x = Taylor1(x0, P)
    @btime jetcoeffs!($f, $t, $x, nothing)

    # TaylorDiff test
    td = TaylorScalar{P}(t0, one(t0))
    @btime my_jetcoeffs($f, $td, $x0, nothing)

    result = my_jetcoeffs(f, td, x0, nothing)
    @assert x.coeffs ≈ collect(flatten(result))
end

function array_test()
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end
    u0 = [1.0; 0.0; 0.0]
    t0 = 0.0
    P = 6

    # TaylorIntegration test
    t = t0 + Taylor1(typeof(t0), P)
    u = [Taylor1(x, P) for x in u0]
    du = similar(u)
    uaux = similar(u)
    @btime jetcoeffs!($lorenz, $t, $u, $du, $uaux, nothing)

    # TaylorDiff test
    td = TaylorScalar{P}(t0, one(t0))
    @btime my_jetcoeffs!($lorenz, $td, $u0, nothing)
    result = my_jetcoeffs!(lorenz, td, u0, nothing)
    for i in eachindex(u)
        @assert u[i].coeffs ≈ collect(flatten(result[i]))
    end
end
