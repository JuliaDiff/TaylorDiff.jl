module TaylorDiffZygoteExt

using TaylorDiff
import Zygote: @adjoint, Numeric, _dual_safearg, ZygoteRuleConfig
using ChainRulesCore: @opt_out

# Zygote can't infer this constructor function
# defining rrule for this doesn't seem to work for Zygote
# so need to use @adjoint
@adjoint TaylorScalar{P}(t::TaylorScalar{T, Q}) where {T, P, Q} = TaylorScalar{P}(t),
x̄ -> (TaylorScalar{Q}(x̄),)

# Zygote will try to use ForwardDiff to compute broadcast functions
# However, TaylorScalar is not dual safe, so we opt out of this
_dual_safearg(::Numeric{<:TaylorScalar}) = false

# Zygote has a rule for literal power, need to opt out of this
@opt_out rrule(
    ::ZygoteRuleConfig, ::typeof(Base.literal_pow), ::typeof(^), x::TaylorScalar, ::Val{p}
) where {p}

end
