module Credit


using JuMP, Ipopt, Distributions, Flux
using Flux.Data: DataLoader
using Plots
using Statistics
using Convex, SCS, Random
using LinearAlgebra


export AbstractCredit

abstract type AbstractCredit end

@with_kw struct Creditrisk <: AbstractCredit
    d::Int = 5
    m::Int = 10
    u::Float64 = 10.0^2
    ρ::Float64 = 2.0
    α::Float64 = 0.05
    β::Float64 = (1-α)^(1/m)
    lb::Float64 = 0.0
    ub::Float64 = 10.0
end


d = 5  # num of obilgor
alpha = 0.95  # CI for VaR 
w = 250  # Threshold VaR 
lower_bound = -2.0
upper_bound = 2.0
q = collect(range(0, stop=10, length=d))
r = collect(range(0.02, stop=0.12, length=d))
n_samples = 10000  # num of MC



function sample_x(lower_bound, upper_bound, num_samples_x)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 


function global_xi(seed)
    # Random.seed!(seed)
    μ = collect(range(0.5, stop=2, length=d))
    σ = 0.5 .* μ
    corr = 0.2
    Σ = Diagonal(σ .^ 2) + fill(corr, d, d)

    ε = 1e-5
    Σ += ε * I

    η = MvNormal(μ, Σ)
    u = collect(range(1, stop=4, length=d))
    ξ = exp.(rand(η) .- u)
    return ξ
end


function cc_g(x,sample_xi)
   return dot(sample_xi, x)-w
end


function neurconst(x::Vector, trained_nn)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
    end
    return x_val[1]
end 


function credit_risk_opt(trained_nn,params)
    opt_model = Model(Ipopt.Optimizer)
    set_silent(opt_model)
    @variable(opt_model, lower_bound <= x[1:d] <= upper_bound) 
    @objective(opt_model, Min, (-sum(q[i]*r[i]*x[i] for i in 1:d) / sum(q)))
    @constraint(opt_model, sum(q[i] * x[i] for i in 1:d) == sum(q))
    for i in 1:d
        @constraint(opt_model, q[i] * x[i] <= 0.20*sum(q))
    end
    @operator(opt_model, new_const, d, (x...) -> neurconst(collect(x),trained_nn))
    @constraint(opt_model, new_const(x...) <= 0)
    optimize!(opt_model)
    if !is_solved_and_feasible(opt_model; allow_almost = true)
        # @show(termination_status(opt_model))
        # @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), -objective_value(opt_model)
end



end



