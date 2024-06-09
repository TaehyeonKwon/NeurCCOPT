module Ordieres

include("data_generation.jl")

using Random, Distributions, JuMP, Ipopt
using .DataGeneration: create_dataset, normalize, split_dataset

export AbstractNorm, NeuralModel

abstract type AbstractNorm end

@with_kw struct Nonconvex <: AbstractNorm
    d::Int = 5
    m::Int = 10
    u::Float64 = 10.0^2
    ρ::Float64 = 2.0
    α::Float64 = 0.05
    β::Float64 = (1-α)^(1/m)
    lb::Float64 = 0.0
    ub::Float64 = 10.0
end


# Design for nonconvex problem
function sample_x(lower_bound, upper_bound, num_samples_x, d)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 

function generate_sample(seed)
    Random.seed!(seed)
    ξ1_dist = Normal(0, 3)
    ξ2_dist = Normal(0, 144)
    return [rand(ξ1_dist), rand(ξ2_dist)]
end

function cc_g(x, y, ξ)
    return 0.25 * x^4 - (1/3) * x^3 - x^2 + 0.2 * x - 19.5 + ξ[1] + ξ[2]-y 
end




function neurconst(x::Vector, trained_nn, Y_max, Y_min)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
    end
    y_pred_normalized = x_val[1]
    y_pred = y_pred_normalized * (Y_max - Y_min) + Y_min
    return y_pred
end 

ξ1_dist = Normal(0, 3)
ξ2_dist = Normal(0, 144)


function NeuralModel(P, trained_nn, params)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, x >= 0)
    @variable(model, y)
    @objective(model, Min, y)
    @operator(model, new_const, params[:d], (x...) -> neurconst(collect(x), trained_nn, params[:Y_max], params[:Y_min]))
    @constraint(model, (new_const(x...) * (params[:Y_max]-params[:Y_min]) + params[:Y_min]) <= params[:epsilon])
    optimize!(model)
    
    if !is_solved_and_feasible(model; allow_almost = true)
        @show termination_status(model)
        println("Warning: Unable to find a feasible and/or optimal solution of the embedded model")
    end
    
    return value(x), value(y)
end


end
