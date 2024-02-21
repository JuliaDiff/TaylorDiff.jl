using TaylorDiff
using TaylorSeries
using TaylorIntegration: jetcoeffs!
using BenchmarkTools

"""
No magic, just a type-stable way to generate a new object since TaylorScalar is immutable
"""
function update_coefficient(x::TaylorScalar{T, N}, index::Integer, value::T) where {T, N}
    return TaylorScalar(ntuple(i -> (i == index ? value : x.value[i]), Val{N}()))
end

"""
Computes the taylor integration of order N - 1, i.e. N = order + 1

eqsdiff: RHS
t: constructed by TaylorScalar{T, N}(t0, 1), which means unit perturbation
x0: initial value
"""
function jetcoeffs_taylordiff(eqsdiff::Function, t::TaylorScalar{T, N}, x0::U, params) where
    {T<:Real, U<:Number, N}
    x = TaylorScalar{U, N}(x0) # x.values[1] is defined, others are 0
    for index in 1:N-1 # computes x.values[index + 1]
        f = eqsdiff(x, params, t)
        df = TaylorDiff.extract_derivative(f, index)
        x = update_coefficient(x, index + 1, df)
    end
    x
end

"""
Computes the taylor integration of order N - 1, i.e. N = order + 1

eqsdiff!: RHS, in non-allocation form
t: constructed by TaylorScalar{T, N}(t0, 1), which means unit perturbation
x0: initial value
"""
function jetcoeffs_array_taylordiff(eqsdiff!::Function, t::TaylorScalar{T, N}, x0::AbstractArray{U, D}, params) where
    {T<:Real, U<:Number, N, D}
    x = map(TaylorScalar{U, N}, x0) # x.values[1] is defined, others are 0
    f = similar(x)
    for index in 1:N-1 # computes x.values[index + 1]
        eqsdiff!(f, x, params, t)
        df = TaylorDiff.extract_derivative.(f, index)
        x = update_coefficient.(x, index + 1, df)
    end
    x
end

"""
In TaylorDiff.jl, the polynomial coefficients are just the n-th order derivatives,
not normalized by n!. So to compare with TaylorSeries.jl, one need to normalize
"""
function normalize_taylordiff_coeffs(t::TaylorScalar)
    return [x / factorial(i - 1) for (i, x) in enumerate(t.value)]
end

function scalar_test()
    rhs(x, p, t) = x * x

    x0 = 0.1
    t0 = 0.0
    order = 6
    N = 7 # N = order + 1

    # TaylorIntegration test
    t = t0 + Taylor1(typeof(t0), order)
    x = Taylor1(x0, order)
    @btime jetcoeffs!($rhs, $t, $x, nothing)

    # TaylorDiff test
    td = TaylorScalar{typeof(t0), N}(t0, one(t0))
    @btime jetcoeffs_taylordiff($rhs, $td, $x0, nothing)

    result = jetcoeffs_taylordiff(rhs, td, x0, nothing)
    normalized = normalize_taylordiff_coeffs(result)
    @assert x.coeffs ≈ normalized
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
    order = 6
    N = 7
    # TaylorIntegration test
    t = t0 + Taylor1(typeof(t0), order)
    u = [Taylor1(x, order) for x in u0]
    du = similar(u)
    uaux = similar(u)
    @btime jetcoeffs!($lorenz, $t, $u, $du, $uaux, nothing)

    # TaylorDiff test
    td = TaylorScalar{typeof(t0), N}(t0, one(t0))
    @btime jetcoeffs_array_taylordiff($lorenz, $td, $u0, nothing)
    result = jetcoeffs_array_taylordiff(lorenz, td, u0, nothing)
    normalized = normalize_taylordiff_coeffs.(result)
    for i in eachindex(u)
        @assert u[i].coeffs ≈ normalized[i]
    end
end
