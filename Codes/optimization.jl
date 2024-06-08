module Optimization

include("utils.jl")
include("problems/hong.jl")
using .Hong: AbstractCCP, NormCCP, NeuralModel

export iterative_retraining

function iterative_retraining(P, model, X_train, Y_train, params)
    # feasibility_train = cc_feasibility(Y_train)  # feasibility_train 초기화
	quantile_values = []
	solutions = []
        feasibility = []
    for iteration in 1:params[:iterations]
        @assert model_validation(model, params[:lower_bound], params[:upper_bound], params[:d])
        x_star_jump, optimal_value = NeuralModel(P, model, params)
        # println("x_star: ",x_star_jump)
        feasi_quantile = quantile(x_star_jump, params[:seed], params[:N], params[:d], params[:m], params[:alpha], params[:case_type])
        # plot_quantile_predictions(X_train, Y_train, model, iteration)
        # println(feasi_quantile)

        push!(quantile_values, feasi_quantile)
        push!(solutions, x_star_jump)
        push!(feasibility, feasi_quantile <= 0)
	if feasi_quantile > 0
           #  println("Iteration $iteration: Infeasible solution found, x* = $x_star_jump")
            for k in 1:params[:K]
                bar_x = rand(Uniform(params[:lower_bound], params[:upper_bound]), params[:d])
                x_k = params[:theta] * x_star_jump + (1 - params[:theta]) * bar_x
                push!(X_train, x_k)
                y_k = quantile(x_k, params[:seed], params[:N], params[:d], params[:m], params[:alpha], params[:case_type])
                if y_k>params[:Y_max]
                    params[:Y_max] = y_k
                end
                if y_k<params[:Y_min]
                    params[:Y_min] = y_k
                end
                y_k_normalized = (y_k - params[:Y_min]) / (params[:Y_max] - params[:Y_min])
                push!(Y_train, y_k_normalized)
                # feasibility_k = cc_feasibility([y_k])
                # push!(feasibility_train, feasibility_k[1])
            end
            X_train_updated = hcat(X_train...)
            Y_train_updated = hcat(Y_train...)
            # feasibility_train_updated = convert(Matrix{Float64}, hcat(feasibility_train...))
            # train_dataset = DataLoader((X_train_updated, Y_train_updated, feasibility_train_updated), batchsize = params[:batch_size], shuffle = true)           
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
