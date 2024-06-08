using CSV, DataFrames, Random
using IterTools: product

include("data_generation.jl")
include("model_training.jl")
include("optimization.jl")
include("utils.jl")
include("problems/hong.jl")

using .DataGeneration: create_dataset, normalize, split_dataset
using .ModelTraining: prepare_train_dataset, train_NN
using .Optimization: iterative_retraining
using .Hong: NormCCP
using .CCPParameters: setup_parameters

if !isdir("results")
    mkdir("results")
end


fixed_params = setup_parameters()

param_ranges = Dict(
    :N => [10,100,1000],
    :num_samples_x => [30, 100, 500, 1000],
    :epsilon => 0:10,
    :K => [10, 30, 50]
)


combinations = product(values(param_ranges)...)


function run_experiment(params)
    X, Y, Feasibility = create_dataset(params)
    Y_normalized, Y_min, Y_max = normalize(Y)
    params[:Y_min] = Y_min
    params[:Y_max] = Y_max

    X_train, X_test, Y_train, Y_test, feasibility_train, feasibility_test = split_dataset(X, Y_normalized, Feasibility)
    
    train_dataset = prepare_train_dataset(X_train, Y_train, feasibility_train, params)
    nn_model = train_NN(train_dataset, params)
    P = NormCCP()
    
    quantile_values, solutions, feasibility = iterative_retraining(P, nn_model, X_train, Y_train, params)
    
    return quantile_values, solutions, feasibility
end

i = 0
for comb in combinations
    global i += 1
    println("=====================Starting combination $i=======================")

    current_params = deepcopy(fixed_params)
    keys_list = keys(param_ranges)
    for (i, key) in enumerate(keys_list)
        current_params[key] = comb[i]
    end
    
    quantile_values, x_solutions, feasibility = run_experiment(current_params)
    
    result_df = DataFrame()
    for k in 1:length(quantile_values)
        x_solution_str = join(x_solutions[k], ", ")
        quantile_value = quantile_values[k]
        is_feasible = feasibility[k]
        append!(result_df, DataFrame(x_solution = x_solution_str, quantile_value = quantile_value, is_feasible = is_feasible))
    end

    file_name = "results/result_d$(current_params[:d])_m$(current_params[:m])_case$(current_params[:case_type])_N$(current_params[:N])_samples$(current_params[:num_samples_x])_epsilon$(current_params[:epsilon])_K$(current_params[:K])_theta$(current_params[:theta]).csv"
    CSV.write(file_name, result_df)
    println("Train done! Result Saved!")
end

println("Experiments completed and results saved in 'results' folder.")
