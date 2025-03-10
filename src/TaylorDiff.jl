module TaylorDiff

"""
    TaylorDiff.can_taylorize(V::Type)

Determines whether the type V is allowed as the scalar type in a
Dual. By default, only `<:Number` types are allowed.
"""
can_taylorize(::Type{<:Number}) = true
can_taylorize(::Type) = false

@noinline function throw_cannot_taylorize(V::Type)
    throw(ArgumentError("Cannot create a Taylor polynomial over scalar type $V." *
                        " If the type behaves as a scalar, define TaylorDiff.can_taylorize(::Type{$V}) = true."))
end

include("scalar.jl")
include("array.jl")
include("utils.jl")
include("primitive.jl")
include("codegen.jl")
include("derivative.jl")
include("chainrules.jl")

end
