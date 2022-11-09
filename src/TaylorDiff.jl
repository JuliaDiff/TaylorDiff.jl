module TaylorDiff

using Zygote
import Base: +, -, *, /, >, <, >=, <=, ==, sin, cos, exp, zero, one

export derivative

include("scalar.jl")
include("vector.jl")
include("derivative.jl")

end
