module Hong
  
include("../utils.jl")
using UnPack


export HongProblem, sample_x, global_xi, cc_g, neurconst, norm_opt


function sample_x(params)
    return [rand(Uniform(params[:lower_bound], params[:upper_bound]), params[:d]) for _ in 1:params[:num_samples_x]]
end 

function global_xi(seed,params)
    Random.seed!(Int(seed))
    if params[:case_type] == 0
        return rand(Normal(0, 1), params[:d], params[:m])
    else 
        means = [j / params[:d] for j in 1:params[:d]]
        cov_matrix = fill(0.05, params[:d], params[:d])
        cov_matrix[diagind(cov_matrix)] .= 1.0
        return rand(MvNormal(means, cov_matrix), params[:m])
    end
end


function cc_g(x, sampled_xi)
    return maximum((dot(x.^2, sampled_xi[:, i].^2) - 100) for i in 1:size(sampled_xi, 2))
end

# function neurconst(x::Vector, trained_nn,params)
#     x_val = copy(x)
#     for i in 1:length(trained_nn)
#         x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
#     end
#     y_nomalize = x_val[1] * (params[:Y_max]-params[:Y_min]) + params[:Y_min]
#     return y_nomalize
# end 

function neurconst(x::Vector, trained_nn, params)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i](x_val)
    end
    y_normalized = x_val[1] * (params[:Y_max] - params[:Y_min]) + params[:Y_min]
    return y_normalized
end


struct HongProblem
    trained_nn::Chain
    params::Dict{Symbol, Any}

    function HongProblem(trained_nn,params)
        new(trained_nn, params)
    end
end


# function norm_problem(problem::HongProblem)
#     model = Model(Ipopt.Optimizer)
#     set_silent(model)
#     @variable(model, problem.params[:lower_bound] <= x[1:problem.params[:d]] <= problem.params[:upper_bound])
#     @objective(model, Min, -sum(x))
#     @operator(model, new_const, problem.params[:d], (x...) -> neurconst(collect(x), problem.trained_nn, problem.params))
#     @constraint(model, new_const(x...) <= 0)
#     optimize!(model)
#     if !is_solved_and_feasible(model; allow_almost = true)
#         @show termination_status(model)
#         @warn("Unable to find a feasible and/or optimal solution of the embedded model")
#     end
#     return value.(x), objective_value(model)
# end
  

function norm_problem(problem::HongProblem)
    model = Model(Ipopt.Optimizer)
    set_silent(model)   
    @variable(model, problem.params[:lower_bound] <= x[1:problem.params[:d]] <= problem.params[:upper_bound]) 
    @objective(model, Min,0)
    @operator(model, new_const, problem.params[:d], (x...) -> neurconst(collect(x), problem.trained_nn, problem.params))
    @constraint(model, new_const(x...) <= 0)
    @constraint(model, sum(x) >= (1*problem.params[:d]))
    optimize!(model)
    if !is_solved_and_feasible(model; allow_almost = true)
        @show(termination_status(model))
        @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), objective_value(model)
end

end
