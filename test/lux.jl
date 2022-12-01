
using Lux, Random

@testset "Lux forward evaluation" begin
    # Construct the layer
    model = Chain(Dense(2, 16, Lux.relu), Dense(16, 1))
    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    # Parameter and State Variables
    ps, st = Lux.setup(rng, model)
    # Dummy Input
    x = TaylorVector([1., 1.], [1., 0.])
    # Run the model
    y, st = Lux.apply(model, x, ps, st)
end

@testset "Lux gradient" begin
    # # Gradients
    # ## Pullback API to capture change in state
    # (l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
    # gs = pb((one.(l), nothing))[1]

    # # Optimization
    # st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
    # st_opt, ps = Optimisers.update(st_opt, ps, gs)
end
