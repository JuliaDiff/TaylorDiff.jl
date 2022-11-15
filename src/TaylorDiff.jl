module TaylorDiff

using Zygote
import Base: +, -, *, /, >, <, >=, <=, ==, sin, cos, exp, zero, one

export derivative, Taylor, TaylorVector

include("scalar.jl")
include("vector.jl")
include("derivative.jl")

end
