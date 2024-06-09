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




function sample_x(lower_bound, upper_bound, num_samples_x, d)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 

function global_xi(params)
    mu = collect(range(0.5, stop=2, length=params[:d]))
    sigma = fill(0.2, params[:d], params[:d]) .+ (0.8 * Diagonal(ones(params[:d]))) 
    u = collect(range(1, stop=4, length=params[:d]))
    M = MvNormal(mu, sqrt.(sigma) ./ 2)  
    Xi = rand(M)  
    return Xi
end

function cc_g(x,params)
    return (exp.(global_xi(params[:d])) .- params[:upper_bound]).*x - 250
end


function neurconst(x::Vector, trained_nn)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
    end
    return x_val[1]
end 




function solve_credit_risk_problem(trained_nn, params)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, lower_bound <= x[i=1:d] <= upper_bound)
    @objective(model, Min, -sum(q[i] * r[i] * x[i] for i in 1:d) / sum(q))
    @constraint(model, sum(q[i] * x[i] for i in 1:d) == sum(q))
    for i in 1:d
        @constraint(model, q[i] * x[i] <= 0.20 * sum(q))
    end
    @operator(model, new_const, d, (x...) -> neurconst(collect(x), trained_nn))
    @constraint(model, new_const(x...) <= 0)
    optimize!(model)
    if !is_solved_and_feasible(model; allow_almost = true)
        # @show termination_status(model)
        # @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), objective_value(model)
end
end



