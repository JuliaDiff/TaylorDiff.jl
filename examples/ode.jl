using ADTypes
using DifferentiationInterface
using ModelingToolkit, DifferentialEquations
using TaylorDiff, ForwardDiff
using Enzyme, Zygote, ReverseDiff
using SciMLSensitivity

@parameters a
@variables t x1(t)
D = Differential(t)
states = [x1]
parameters = [a]

@named pre_model = ODESystem([D(x1) ~ a * x1], t, states, parameters)
model = structural_simplify(pre_model)

ic = Dict(x1 => 1.0)
p_true = Dict(a => 2.0)

problem = ODEProblem{true, SciMLBase.FullSpecialize}(model, ic, [0.0, 1.0], p_true)
soln = ModelingToolkit.solve(problem, Tsit5(), abstol = 1e-12, reltol = 1e-12)
display(soln(0.5, idxs = [x1]))

function different_time(new_ic, new_params, new_t)
    #newprob = ODEProblem{true, SciMLBase.FullSpecialize}(model, new_ic, [0.0, new_t*2], new_params)
    #newprob = remake(problem, u0=new_ic, tspan = [0.0, new_t], p = new_params)
    newprob = remake(problem, u0 = new_ic, tspan = [0.0, new_t], p = new_params)
    newprob = remake(newprob, u0 = typeof(new_t).(newprob.u0))
    new_soln = ModelingToolkit.solve(newprob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    return (soln(new_t, idxs = [x1]))
end

function just_t(new_t)
    return different_time(ic, p_true, new_t)[1]
end
display(different_time(ic, p_true, 2e-5))
display(just_t(0.5))

#display(ForwardDiff.derivative(just_t,1.0))
display(TaylorDiff.derivative(just_t, 1.0, 1))  #isnan error
#display(value_and_gradient(just_t, AutoForwardDiff(), 1.0))
#display(value_and_gradient(just_t, AutoReverseDiff(), 1.0))
#display(value_and_gradient(just_t, AutoEnzyme(Enzyme.Reverse), 1.0))
#display(value_and_gradient(just_t, AutoEnzyme(Enzyme.Forward), 1.0))
#display(value_and_gradient(just_t, AutoZygote(), 1.0))
