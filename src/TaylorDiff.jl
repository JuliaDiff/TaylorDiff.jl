module TaylorDiff

using Zygote
import Base: abs, abs2
import Base: exp, exp2, exp10, expm1, log, log2, log10, log1p, inv, sqrt, cbrt
import Base: sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch
import Base: asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth, asech, acsch, sinc, cosc
import Base: zero, one
import Base: +, -, *, /, \, ^, >, <, >=, <=, ==
import Base: hypot, max, min

export derivative, Taylor, TaylorVector

include("scalar.jl")
include("vector.jl")
include("derivative.jl")

end
