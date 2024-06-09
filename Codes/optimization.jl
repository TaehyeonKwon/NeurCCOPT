module Optimization

include("utils.jl")
include("problems/hong2.jl")
using .Hong: HongProblem, sample_x as hong_sample_x, global_xi as hong_global_xi, cc_g as hong_cc_g, neurconst as hong_neurconst, norm_opt as hong_norm_opt

export iterative_retraining

function iterative_retraining(problem_instance, model, X_train, Y_train, params, norm_opt_func, global_xi_func, cc_g_func)
	quantile_values = []
	solutions = []
    feasibility = []
    for iteration in 1:params[:iterations]
        @assert model_validation(model, params[:lower_bound], params[:upper_bound], params[:d])
        x_star_jump, optimal_value = norm_opt_func(problem_instance)
        # println("x_star: ",x_star_jump)
        # feasi_quantile = compute_quantile(x_star_jump, params, global_xi, cc_g) 
        feasi_quantile = compute_quantile(x_star_jump, params, global_xi_func, cc_g_func)
        push!(quantile_values, feasi_quantile)
        push!(solutions, x_star_jump)
        push!(feasibility, feasi_quantile <= 0)
	if feasi_quantile > 0
           #  println("Iteration $iteration: Infeasible solution found, x* = $x_star_jump")
            for k in 1:params[:K]
                bar_x = rand(Uniform(params[:lower_bound], params[:upper_bound]), params[:d])
                x_k = params[:theta] * x_star_jump + (1 - params[:theta]) * bar_x
                push!(X_train, x_k)
                y_k = compute_quantile(x_k, params, global_xi_func, cc_g_func)
                push!(Y_train, y_k)
            end
            X_train_updated = hcat(X_train...)
            Y_train_updated = hcat(Y_train...)
            train_dataset = DataLoader((X_train_updated, Y_train_updated), batchsize = params[:batch_size], shuffle = true)           
            model = train_NN(train_dataset, params)
        else
            println("Iteration $iteration: Feasible solution found, x* = $x_star_jump. No retraining required.")
            break
        end
    end
    return quantile_values, solutions, feasibility
end

end # module
