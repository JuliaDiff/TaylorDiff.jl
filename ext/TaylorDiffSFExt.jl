module TaylorDiffSFExt
using TaylorDiff, SpecialFunctions
# Names used by the code that `define_unary_function` generates in this module
using TaylorDiff: TaylorScalar, value, partials, raise, raiseinv
using TaylorDiff.ChainRulesCore: frule, NoTangent, zero_tangent

for func in (erf, erfc, erfcinv, erfcx, erfi)
    TaylorDiff.define_unary_function(func, TaylorDiffSFExt)
end

end
