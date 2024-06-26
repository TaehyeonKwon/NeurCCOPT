using CSV, DataFrames, Random
using IterTools: product

include("optimization.jl")
include("utils.jl")
include("problems/hong.jl")
include("problems/credit_risk.jl")
# include("problems/nonconvex.jl")

using .Optimization: iterative_retraining
using .CCPParameters: setup_parameters
using .Hong: HongProblem, sample_x as hong_sample_x, global_xi as hong_global_xi, cc_g as hong_cc_g, neurconst as hong_neurconst, norm_problem
using .Credit: CreditProblem, sample_x as credit_sample_x, global_xi as credit_global_xi, cc_g as credit_cc_g, neurconst as credit_neurconst, credit_problem

if !isdir("results")
    mkdir("results")
end


problems = Dict(
    1 => (HongProblem, hong_sample_x, hong_global_xi, hong_cc_g, hong_neurconst, norm_problem),
    2 => (CreditProblem, credit_sample_x, credit_global_xi, credit_cc_g, credit_neurconst, credit_problem)
)

# Problem indicator
indicator = 1
problem_info = problems[indicator]
fixed_params = setup_parameters(indicator)

param_ranges = Dict(
#    :N => [100,1000],
    :num_samples_x => [100],
    :K => [50],
    # :theta => 0.8:0.1:0.9
    :theta => 0.9
)


combinations = product(values(param_ranges)...)


function run_experiment(params,problem_info)   
    # Receive problem instance
    problem_constructor, sample_x_func, global_xi_func, cc_g_func, neurconst_func, opt_problem = problem_info
    # Create initial training dataset
    X, Y = create_dataset(params, sample_x_func,global_xi_func, cc_g_func)
    # Normalized generated initial data for neural net training
    X_scaled, Y_scaled = min_max_scaling(X, Y, params)
    # Split data set; for initial neural net training, for update on iterative learning 
    X_train, X_test, Y_train, Y_test = split_dataset(X_scaled, Y_scaled)
    X_train_original, X_test_original, Y_train_original, Y_test_original = split_dataset(X, Y)
    # Prepare dataset & neural net train 
    initial_train_dataset = prepare_train_dataset(X_train, Y_train, params)
    initial_nn_model = train_NN(initial_train_dataset, params)
    # Generate problem instance for iterative learning
    problem_instance = problem_constructor(initial_nn_model, params) 
    quantile_values, solutions, feasibility, optimal_value = iterative_retraining(problem_instance, initial_nn_model, X_train_original, Y_train_original, params, opt_problem, global_xi_func, cc_g_func)    
    return quantile_values, solutions, feasibility, optimal_value
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
    quantile_values, x_solutions, feasibility, optimal_value = run_experiment(current_params,problem_info)
    
    result_df = DataFrame()
    for k in 1:length(quantile_values)
        x_solution_str = join(x_solutions[k], ", ")
        quantile_value = quantile_values[k]
        is_feasible = feasibility[k]
        append!(result_df, DataFrame(x_solution = x_solution_str, quantile_value = quantile_value, is_feasible = is_feasible))
    end
    println("optimal value: ",optimal_value)
    file_name = "results/result_problem$(indicator)_d$(current_params[:d])_N$(current_params[:N])_K$(current_params[:K])_theta$(current_params[:theta]).csv"
    CSV.write(file_name, result_df)
    println("Train done! Result Saved!")
    error("stop!")
end

# alpha set and obj_value set should be passed
# plot_feasible_frontier(alpha, obj_value)
println("Experiments completed and results saved in 'results' folder.")
