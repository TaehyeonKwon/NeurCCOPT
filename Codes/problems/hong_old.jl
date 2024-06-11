module Hong
  
using Parameters
using JuMP, Ipopt
using UnPack

export AbstractNorm, NormCCP, NeuralModel

abstract type AbstractNorm end

@with_kw struct NormCCP <: AbstractNorm
    d::Int = 5
    m::Int = 10
    u::Float64 = 10.0^2
    ρ::Float64 = 2.0
    α::Float64 = 0.05
    β::Float64 = (1-α)^(1/m)
    lb::Float64 = 0.0
    ub::Float64 = 10.0
end

function neurconst(x::Vector, trained_nn, Y_max, Y_min)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
    end
    return x_val[1]
end 


# function NeuralModel(P, trained_nn, params)
#     # @unpack d,lb,ub,ϵ = P
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, params[:lower_bound] <= x[1:params[:d]] <= params[:upper_bound])
#     @objective(model, Min, -sum(x))
#     @operator(model, new_const, params[:d], (x...) -> neurconst(collect(x), trained_nn, params[:Y_max], params[:Y_min]))
#     @constraint(model, (new_const(x...) * (params[:Y_max]-params[:Y_min]) + params[:Y_min]) <= params[:epsilon])
#     optimize!(model)
#     if !is_solved_and_feasible(model; allow_almost = true)
#         # @show termination_status(model)
#         # @warn("Unable to find a feasible and/or optimal solution of the embedded model")
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
    # @constraint(model, (new_const(x...) * (params[:Y_max]-params[:Y_min]) + params[:Y_min]) <= params[:epsilon])
    @constraint(model, new_const(x...) <= params[:epsilon])
    @constraint(model, sum(x) >= (1*params[:d]))
    optimize!(model)
    if !is_solved_and_feasible(model; allow_almost = true)
       #  @show(termination_status(model))
       #  @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), objective_value(model)
end

end