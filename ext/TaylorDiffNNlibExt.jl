module TaylorDiffNNlibExt

using TaylorDiff
import NNlib: oftf
import NNlib: sigmoid_fast, tanh_fast, rrelu, leakyrelu

@inline sigmoid_fast(t::TaylorScalar) = one(t) / (one(t) + exp(-t))

@inline tanh_fast(t::TaylorScalar) = tanh(t)

@inline function rrelu(t::TaylorScalar{T, N},
        l = oftf(t, 1 / 8),
        u = oftf(t, 1 / 3)) where {T, N}
    a = (u - l) * rand(float(T)) + l
    return leakyrelu(t, a)
end

end
