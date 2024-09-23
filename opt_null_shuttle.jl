ENV["PYTHON"] = "/Users/richardr2926/.pyenv/versions/3.10.4/bin/python"

using DrWatson

@quickactivate "FNO-CIG"

using NLopt
using Zygote
using Pkg
using Conda

Conda.add("matplotlib")
Conda.add("numpy=1.24")
Pkg.build("PyCall")

include("hypothesis.jl")
include("objective.jl")

function vHp(m, v, φ::Objective)
    ∂φ∂m = Zygote.gradient(m -> φ(m), m)
    ∂φ∂m∂m = Zygote.gradient(m -> φ(m), m)
end

function run_optimizer(ψ::Hypothesis, φ::Objective; max_shuttles::Int = 5, max_steps::Int = 20, tol::Float64 = 1e-5)
    # ψ: Hypothesis to minimize
    # φ: Objective to satisfy
    # max_shuttles: Maximum no of shuttles
    # tol: Tolerance for stopping criteria

    m = get_starting_model(φ)

    for shuttle = 1:max_shuttles
        Δm, ∂φ∂m = init(ψ, φ, m)

        function objective!(Δm, ∂ψ∂Δm)
            
            # Compute the value of hypothesis at perturbed point
            δm̂ = Δm / norm(Δm)
            α = (-2 * g0 * δm̂) / (-δm̂.T * Hδm̂)
            Δm′ = α * δm̂

            value, ∂ψ∂Δm′ = ψ(m + Δm′), Zygote.gradient(xΔm′ -> ψ(m + Δm′), Δm′)

            Hδm̂ = vHp(m, δm̂, φ) # Compute Hessian product of φ at m with vector δm̂
            δm̂THδm̂ = δm̂.T * Hδm̂

            ∂α∂δm̂ = -(2 * ∂φ∂m / δm̂THδm̂) + 4 * Hδm̂ * δm̂ * (∂φ∂m * δm̂ / δm̂THδm̂)
            ∂α∂Δm = (1 / norm(Δm)) * ∂α∂δm̂ - (Δm / norm(Δm)^3) * ∂α∂δm̂.T * Δm
            _∂ψ∂Δm = ∂α∂Δm * (∂ψ∂Δm′.T * δm̂) + α * ((1 / norm(Δm)) * ∂ψ∂Δm′ - (Δm / norm(Δm)^3) * ∂ψ∂Δm′.T * Δm)
            
            ∂ψ∂Δm[1:end] = vec(_∂ψ∂Δm)
            return value
        end

        # Test the first gradient
        g = zeros(prod(model0.n))
        f0 = objective!(vec(model0.m), g)

        # Squared slowness bounds
        mmax = (1.3f0) .^ (-2)
        mmin = (6.5f0) .^ (-2)

        opt = Opt(:LD_LBFGS, prod(model0.n))
        opt.lower_bounds = mmin
        opt.upper_bounds = mmax

        opt.min_objective = objective!
        opt.maxeval = max_steps

        @time (minf, minx, ret) = optimize(opt, model0.m[:])
        Δm = minx
        m += Δm # TODO: Line search again w.r.t objective function
    end
    return ∆m
end

function main()
    # Define the hypothesis and objective
    ψ = DummyHypothesis()
    φ = FocusImageGatechObjective()

    # Run the optimization
    optimized_model = run_optimizer(ψ, φ, max_shuttles=1, max_steps=1)
end

main()
