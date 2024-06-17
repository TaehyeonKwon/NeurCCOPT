module Optimization

include("utils.jl")
# include("problems/hong.jl")
# using .Hong: HongProblem, sample_x as hong_sample_x, global_xi as hong_global_xi, cc_g as hong_cc_g, neurconst as hong_neurconst, norm_opt as hong_norm_opt

export iterative_retraining

function iterative_retraining(problem_instance, model, X_train_original, Y_train_original, params, opt_problem, global_xi_func, cc_g_func)
	quantile_values = []
	solutions = []
    feasibility = []
    obj_value = 0
    for iteration in 1:params[:iterations]
        @assert model_validation(model, params[:lower_bound], params[:upper_bound], params[:d])
        x_star_jump, optimal_value = opt_problem(problem_instance)
        quantile_value = compute_quantile(x_star_jump, params, global_xi_func, cc_g_func)
        is_feasible = check_feasibility(x_star_jump, params, cc_g_func, global_xi_func)
        obj_value = optimal_value

        push!(quantile_values, quantile_value)
        push!(solutions, x_star_jump)
        push!(feasibility, is_feasible)
	    
        if !is_feasible
           #  println("Iteration $iteration: Infeasible solution found, x* = $x_star_jump")
            for k in 1:params[:K]
                lb = fill(params[:lower_bound], params[:d])
                ub = fill(params[:upper_bound], params[:d])
                # bar_x = rand(Uniform(params[:lower_bound], params[:upper_bound]), params[:d])
                samples = QuasiMonteCarlo.sample(1, lb, ub, SobolSample())
                bar_x = samples[:, 1]
                x_k = params[:theta] * x_star_jump + (1 - params[:theta]) * bar_x
                push!(X_train_original, x_k)
                y_k = compute_quantile(x_k, params, global_xi_func, cc_g_func)
                push!(Y_train_original, y_k)
            end
            X_normalized, Y_normalized = min_max_scaling(X_train_original, Y_train_original,params)
            train_dataset = DataLoader((hcat(X_normalized...), hcat(Y_normalized...)), batchsize = params[:batch_size], shuffle = true)           
            model = train_NN(train_dataset, params)
        else
            println("Iteration $iteration: Feasible solution found, x* = $x_star_jump. No retraining required.")
            break
        end
    end
    return quantile_values, solutions, feasibility, obj_value
end

end # module
