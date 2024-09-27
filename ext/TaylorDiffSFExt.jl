module TaylorDiffSFExt
using TaylorDiff, SpecialFunctions

for func in (erf, erfc, erfcinv, erfcx, erfi)
    TaylorDiff.define_unary_function(func, TaylorDiffSFExt)
end

end
