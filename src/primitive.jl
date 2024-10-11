import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch, sinpi, cospi
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch
import Base: sinc, cosc
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min
import Base: tail
import Base: convert, promote_rule

Taylor = Union{TaylorScalar, TaylorArray}

@inline value(t::Taylor) = t.value
@inline partials(t::Taylor) = t.partials
@inline @generated extract_derivative(t::Taylor, ::Val{P}) where {P} = :(t.partials[P] *
                                                                         $(factorial(P)))
@inline extract_derivative(a::AbstractArray{<:TaylorScalar}, p) = map(
    t -> extract_derivative(t, p), a)
@inline extract_derivative(_, p) = false
@inline extract_derivative!(result, a::AbstractArray{<:TaylorScalar}, p) = map!(
    t -> extract_derivative(t, p), result, a)

@inline flatten(t::Taylor) = (value(t), partials(t)...)

function promote_rule(::Type{TaylorScalar{T, P}},
        ::Type{S}) where {T, S, P}
    TaylorScalar{promote_type(T, S), P}
end

function (::Type{F})(x::TaylorScalar{T, P}) where {T, P, F <: AbstractFloat}
    F(value(x))
end

# Unary

## Delegated

@inline +(t::TaylorScalar) = t
@inline -(t::TaylorScalar) = TaylorScalar(-value(t), .-partials(t))
@inline sqrt(t::TaylorScalar) = t^0.5
@inline cbrt(t::TaylorScalar) = ^(t, 1 / 3)
@inline inv(t::TaylorScalar) = one(t) / t

for func in (:exp, :expm1, :exp2, :exp10)
    @eval @generated function $func(t::TaylorScalar{T, P}) where {T, P}
        v = [Symbol("v$i") for i in 0:P]
        ex = quote
            $(Expr(:meta, :inline))
            p = value(t)
            f = flatten(t)
            v0 = $($(QuoteNode(func)) == :expm1 ? :(exp(p)) : :($$func(p)))
        end
        for i in 1:P
            push!(ex.args,
                :(
                    $(v[begin + i]) = +($([:($(i - j) * $(v[begin + j]) *
                                             f[begin + $(i - j)])
                                           for j in 0:(i - 1)]...)) / $i
                ))
            if $(QuoteNode(func)) == :exp2
                push!(ex.args, :($(v[begin + i]) *= log(2)))
            elseif $(QuoteNode(func)) == :exp10
                push!(ex.args, :($(v[begin + i]) *= log(10)))
            end
        end
        if $(QuoteNode(func)) == :expm1
            push!(ex.args, :(v0 = expm1(f[1])))
        end
        push!(ex.args, :(TaylorScalar(tuple($(v...)))))
        return :(@inbounds $ex)
    end
end

for func in (:sin, :cos)
    @eval @generated function $func(t::TaylorScalar{T, P}) where {T, P}
        s = [Symbol("s$i") for i in 0:P]
        c = [Symbol("c$i") for i in 0:P]
        ex = quote
            $(Expr(:meta, :inline))
            f = flatten(t)
            s0 = sin(f[1])
            c0 = cos(f[1])
        end
        for i in 1:P
            push!(ex.args,
                :($(s[begin + i]) = +($([:(
                                             $(i - j) * $(c[begin + j]) *
                                             f[begin + $(i - j)]) for j in 0:(i - 1)]...)) /
                                    $i)
            )
            push!(ex.args,
                :($(c[begin + i]) = +($([:(
                                             $(i - j) * $(s[begin + j]) *
                                             f[begin + $(i - j)]) for j in 0:(i - 1)]...)) /
                                    -$i)
            )
        end
        if $(QuoteNode(func)) == :sin
            push!(ex.args, :(TaylorScalar(tuple($(s...)))))
        else
            push!(ex.args, :(TaylorScalar(tuple($(c...)))))
        end
        return :(@inbounds $ex)
    end
end

@inline sinpi(t::TaylorScalar) = sin(π * t)
@inline cospi(t::TaylorScalar) = cos(π * t)

# Binary

## Easy case

@inline +(a::Number, b::TaylorScalar) = TaylorScalar(a + value(b), partials(b))
@inline -(a::Number, b::TaylorScalar) = TaylorScalar(a - value(b), .-partials(b))
@inline *(a::Number, b::TaylorScalar) = TaylorScalar(a * value(b), a .* partials(b))
@inline /(a::Number, b::TaylorScalar) = /(promote(a, b)...)

@inline +(a::TaylorScalar, b::Number) = TaylorScalar(value(a) + b, partials(a))
@inline -(a::TaylorScalar, b::Number) = TaylorScalar(value(a) - b, partials(a))
@inline *(a::TaylorScalar, b::Number) = TaylorScalar(value(a) * b, partials(a) .* b)
@inline /(a::TaylorScalar, b::Number) = TaylorScalar(value(a) / b, partials(a) ./ b)

const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

for op in [:>, :<, :(==), :(>=), :(<=)]
    for R in AMBIGUOUS_TYPES
        @eval @inline $op(a::TaylorScalar, b::$R) = $op(value(a)[1], b)
        @eval @inline $op(a::$R, b::TaylorScalar) = $op(a, value(b)[1])
    end
    @eval @inline $op(a::TaylorScalar, b::TaylorScalar) = $op(value(a)[1], value(b)[1])
end

@inline +(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(
    value(a) + value(b), map(+, partials(a), partials(b)))
@inline -(a::TaylorScalar, b::TaylorScalar) = TaylorScalar(
    value(a) - value(b), map(-, partials(a), partials(b)))

@generated function *(a::TaylorScalar{T, N}, b::TaylorScalar{T, N}) where {T, N}
    return quote
        $(Expr(:meta, :inline))
        va, vb = flatten(a), flatten(b)
        v = tuple($([:(
                         +($([:(va[begin + $j] * vb[begin + $(i - j)]) for j in 0:i]...))
                     ) for i in 0:N]...))
        @inbounds TaylorScalar(v)
    end
end

@generated function /(a::TaylorScalar{T, P}, b::TaylorScalar{T, P}) where {T, P}
    v = [Symbol("v$i") for i in 0:P]
    ex = quote
        $(Expr(:meta, :inline))
        va, vb = flatten(a), flatten(b)
        v0 = va[1] / vb[1]
        b0 = vb[1]
    end
    for i in 1:P
        push!(ex.args,
            :(
                $(v[begin + i]) = (va[begin + $i] -
                                   +($([:($(v[begin + j]) *
                                          vb[begin + $(i - j)])
                                        for j in 0:(i - 1)]...))) / b0
            )
        )
    end
    push!(ex.args, :(TaylorScalar(tuple($(v...)))))
    return :(@inbounds $ex)
end

for R in (Integer, Real)
    @eval @generated function ^(t::TaylorScalar{T, P}, n::S) where {S <: $R, T, P}
        v = [Symbol("v$i") for i in 0:P]
        ex = quote
            $(Expr(:meta, :inline))
            f = flatten(t)
            f0 = f[1]
            v0 = ^(f0, n)
        end
        for i in 1:P
            push!(ex.args,
                :(
                    $(v[begin + i]) = +($([:(
                                               (n * $(i - j) - $j) * $(v[begin + j]) *
                                               f[begin + $(i - j)]
                                           ) for j in 0:(i - 1)]...)) / ($i * f0)
                ))
        end
        push!(ex.args, :(TaylorScalar(tuple($(v...)))))
        return :(@inbounds $ex)
    end
    @eval function ^(a::S, t::TaylorScalar{T, N}) where {S <: $R, T, N}
        exp(t * log(a))
    end
end

^(t::TaylorScalar, s::TaylorScalar) = exp(s * log(t))

@inline function lower(t::TaylorScalar{T, P}) where {T, P}
    s = partials(t)
    TaylorScalar(ntuple(i -> s[i] * i, Val(P)))
end
@inline function higher(t::TaylorScalar{T, P}) where {T, P}
    s = flatten(t)
    ntuple(i -> s[i] / i, Val(P + 1))
end
@inline raise(f, df::TaylorScalar, t) = TaylorScalar(f, higher(lower(t) * df))
@inline raise(f, df::Number, t) = df * t
@inline raiseinv(f, df, t) = TaylorScalar(f, higher(lower(t) / df))
