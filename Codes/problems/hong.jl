module Hong
  
using Parameters
using JuMP, Ipopt
using UnPack

export AbstractCCP, NormCCP, NeuralModel

abstract type AbstractCCP end

@with_kw struct NormCCP <: AbstractCCP
    d::Int = 5
    m::Int = 10
    u::Float64 = 10.0^2
    ρ::Float64 = 2.0
    α::Float64 = 0.05
    β::Float64 = (1-α)^(1/m)
    lb::Float64 = 0.0
    ub::Float64 = 10.0
    ϵ::Int = 10
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


# function NeuralModel(P, trained_nn, Y_max, Y_min)
#     @unpack d,lb,ub,ϵ = P
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, lb <= x[1:d] <= ub)
#     @objective(model, Min, -sum(x))
#     @operator(model, new_const, d, (x...) -> neurconst(collect(x), trained_nn, Y_max, Y_min))
#     @constraint(model, new_const(x...) <= ϵ)
#     optimize!(model)
#     if !is_solved_and_feasible(model; allow_almost = true)
#         @show termination_status(model)
#         @warn("Unable to find a feasible and/or optimal solution of the embedded model")
#     end
#     return value.(x), objective_value(model)
# end
  

function NeuralModel(P, trained_nn, params)
    # @unpack d,lb,ub,ϵ = P
    model = Model(Ipopt.Optimizer)
    set_silent(model)   
    @variable(model, params[:lower_bound] <= x[1:params[:d]] <= params[:upper_bound]) 
    @objective(model, Min,0)
    @operator(model, new_const, params[:d], (x...) -> neurconst(collect(x), trained_nn, params[:Y_max], params[:Y_min]))
    @constraint(model, new_const(x...) <= (params[:epsilon]-params[:Y_min])/(params[:Y_max]-params[:Y_min]))
    @constraint(model, sum(x) >= (1*params[:d]))
    optimize!(model)
    if !is_solved_and_feasible(model; allow_almost = true)
        @show(termination_status(model))
        @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), objective_value(model)
end

end
