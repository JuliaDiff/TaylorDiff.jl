
function derivative(f::Function, x::T, order::Int64) where {T<:Number}
    m = Taylor(x, one(x), zeros(T, order - 1)...)
    result = f(m)
    return getindex(value(result), order + 1)
end

function derivative(f::Function, x::Vector{T}, index::Int64, order::Int64) where {T<:Number}
    nx = [Taylor(val, ind == index ? one(T) : zero(T), zeros(T, order - 1)...) for (ind, val) in enumerate(x)]
    result = f(nx)
    return getindex(value(result), order + 1)
end
