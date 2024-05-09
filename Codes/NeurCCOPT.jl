# This code was implemented Hong et al.(2010) Section 5.1.1 problem
# Paper: https://pubsonline.informs.org/doi/pdf/10.1287/opre.1100.0910

using JuMP, Ipopt, Distributions, Flux
using Flux.Data: DataLoader
using Plots
using Statistics, Random
using LinearAlgebra


# 1. Given_Parameters
d = 5  # Degrees of freedom for ξ_i
alpha = 0.05  # Confidence level
m = 10^2 # number of constraint inside probability
beta = (1 - alpha)^(1/m)
lower_bound = 0.0 # lower bound for x
upper_bound = 10.0 # upper bound for x
case_type = 1 # case type; 0: independent, 1: dependent

if case_type == 0
    quantile_value = Distributions.quantile(Chisq(d), beta)
else
    quantile_value = Distributions.quantile(Chisq(d), 1-alpha)
end


# 2. hyperparameters
N = 10^3 # number of scenario
num_samples_x = 100 # initial generated number of x
epsilon = 100 # threshold for embedded nn constraint
seed = 1 # random seed for data generation
## 2.1 NN model
batch_size = 5
epochs = 30
learning_rate = 0.01
## 2.2 Framework Parameter
iterations = 1000
K = 30  # alternating sample size
theta = 0.9 # convexity parameter


function sample_x(lower_bound, upper_bound, num_samples_x)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 


function generate_sample(seed)
    Random.seed!(seed)
    if case_type == 0
        xi_vector = rand(Normal(0,1), d, m)
        return xi_vector
    else 
        means = [j / d for j in 1:d]  # Means depend linearly on j, scaled by d
        cov_matrix = fill(0.05, d, d)  # Initialize covariance matrix with 0.05 for all elements
        # Use diagind to access and set diagonal elements to 1 (variance of each variable)
        diag_indices = diagind(cov_matrix)
        cov_matrix[diag_indices] .= 1.0
        # Generate m samples of a d-dimensional multivariate normal distribution
        xi_matrix = rand(MvNormal(means, cov_matrix), m)  # m columns each of dimension d
        return xi_matrix  # Returning a d x m matrix where each column is one sample
    end
end


function cc_g(x, sampled_xi)
    constraint_value = 100
    return_value = maximum((dot(x.^2, sampled_xi[:,i].^2) - constraint_value) for i in 1:size(sampled_xi, 2))
    return return_value
end


function quantile(x, seed)
    results = Float64[] 
    for i in 1:N
        sample_xi = generate_sample(seed + i)
        push!(results, cc_g(x, sample_xi))
    end
    sorted_results = sort(results; rev=true)
    index = ceil(Int, alpha * N)
    return sorted_results[index]
end


# Currently working on...
# Problem: There are big differences between pred value (<10) vs actual value (100<)
# Solution: Improve model performance
function plot_quantile_predictions(X_test, Y_test, model, iteration)
    Y_pred = Float64.([model(x)[1] for x in X_test])
    Y_test_vec = vec(Float64.(Y_test))
    min_val = minimum([Y_test_vec; Y_pred]) - 50  
    max_val = maximum([Y_test_vec; Y_pred]) + 50  

    scatterplot = scatter(Y_test_vec, Y_pred,
                          label="Predicted vs Actual",
                          title="Quantile Prediction Accuracy - Iteration $iteration",
                          xlabel="Actual Quantile",
                          ylabel="Predicted Quantile",
                          xlims=(min_val, max_val),
                          ylims=(min_val, max_val),
                          legend=:topleft,
                          size=(600, 600), 
                          msize=6, 
                          mcolor=:blue,  
                          grid=true)  

    plot!(scatterplot, [min_val, max_val], [min_val, max_val],
          label="Ideal Line y = x", line=(:red, :dash, 2), color=:red)


    savepath = "./plots"
    if !isdir(savepath)
        mkdir(savepath)
    end
    savefig(scatterplot, joinpath(savepath, "iteration_$iteration.png"))
end


function create_dataset(lower_bound, upper_bound, num_samples_x, seed)
    X = sample_x(lower_bound, upper_bound, num_samples_x)
    Y = [quantile(x,seed) for x in X]
    return X, Y
end


X, Y = create_dataset(lower_bound, upper_bound, num_samples_x, seed)
println("Dataset generated.")


train_set_end = floor(Int, length(X) * 0.8)

X_train, X_test = X[1:train_set_end], X[train_set_end+1:end]
Y_train, Y_test = Y[1:train_set_end], Y[train_set_end+1:end]

X_train = vec(X_train)
Y_train = vec(Y_train)

train_dataset = DataLoader((hcat(X_train...), hcat(Y_train...)), batchsize=batch_size, shuffle=false)

# model couldn't predict well.
function train_NN(train_dataset)
    Random.seed!(123)
    model = Chain(Dense(d, 3, sigmoid), Dense(3, 1))
    optimizer = ADAM(learning_rate)
    loss(x,y) = Flux.Losses.mse(model(x),y)
    params = Flux.params(model)  
    for epoch in 1:epochs
        for batch in train_dataset
            Flux.train!(loss, params, [batch], optimizer)
        end
        current_loss = mean([loss(first(batch), last(batch)) for batch in train_dataset])
        @info "Epoch: $epoch , Loss:  $current_loss"
    end
    return model
end


function model_output_function_jl(x::Vector{Float64}, model)
    weights_1, bias_1 = model[1].weight, model[1].bias
    weights_2, bias_2 = model[2].weight, model[2].bias
    dense1_output = sigmoid.(weights_1 * x .+ bias_1)
    dense2_output = (weights_2 * dense1_output .+ bias_2)
    return dense2_output[1]  
end

function model_validation(model)
    x=rand(Uniform(lower_bound, upper_bound), d)
    model_output = model(x)
    feasibility = isapprox(model_output[1],model_output_function_jl(x,model), atol=1e-5)
    return feasibility
end 


#### Case 1: original problem ####

# function solve_norm_opt_probelm(lower_bound, upper_bound, alpha,trained_nn)
#     opt_model = Model(Ipopt.Optimizer)
#     set_optimizer_attribute(opt_model, "print_level", 0)
#     function f(x::Vector)
#         x_val = copy(x)
#         for i in 1:length(trained_nn)
#             x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
#         end
#         return x_val[1]
#     end   
#     @variable(opt_model, lower_bound <= x[1:d] <= upper_bound)
#     @objective(opt_model, Min, -sum(x))
#     @operator(opt_model, new_const, d, (x...) -> f(collect(x)))
#     @constraint(opt_model, new_const(x...) <= epsilon)
#     optimize!(opt_model)
#     return value.(x), objective_value(opt_model)
# end


#### Case 2: obj function included to constraint ####

function solve_norm_opt_probelm(lower_bound, upper_bound, alpha,trained_nn)
    opt_model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(opt_model, "print_level", 0)
    function f(x::Vector)
        x_val = copy(x)
        for i in 1:length(trained_nn)
            x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
        end
        return x_val[1]
    end   
    @variable(opt_model, lower_bound <= x[1:d] <= upper_bound) 
    @objective(opt_model, Min,0)
    @operator(opt_model, new_const, d, (x...) -> f(collect(x)))
    @constraint(opt_model, new_const(x...) <= epsilon)
    @constraint(opt_model, sum(x) >= (0*d))
    optimize!(opt_model)
    if !is_solved_and_feasible(opt_model; allow_almost = true)
        @show(termination_status(opt_model))
        @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), objective_value(opt_model)
end



nn_model = train_NN(train_dataset)
println("Initial model training completed!")


function iterative_retraining(iterations, K, theta, model)
    for iteration in 1:iterations 
        @assert model_validation(model)
        x_star_jump,optimal_value = solve_norm_opt_probelm(lower_bound, upper_bound, alpha,model)
        feasi_quantile = quantile(x_star_jump, seed)
        
        plot_quantile_predictions(X_train, Y_train, model, iteration)
        if feasi_quantile > 0
            println("Iteration $iteration: Infeasible solution found, x* = $x_star_jump")
            for k in 1:K
                bar_x = rand(Uniform(lower_bound, upper_bound), d)
                x_k = theta*x_star_jump + (1-theta)*bar_x
                push!(X_train, x_star_jump)
                push!(Y_train, quantile(x_k,seed))
            end
            X_train_updated, Y_train_updated = hcat(X_train...), hcat(Y_train...)
            train_dataset = DataLoader((X_train_updated, Y_train_updated), batchsize=batch_size, shuffle=true)
            model = train_NN(train_dataset)
        else
            println("Iteration $iteration: Feasible solution found, x* = $x_star_jump. No retraining required.")
            break
        end
    end
    return model
end

# Begin the iterative retraining process
retraining_nn_model = iterative_retraining(iterations, K, theta, nn_model)
if case_type == 0
    quantile_value = Distributions.quantile(Chisq(d), beta)
    x_star_closed_form = 10 / sqrt(quantile_value)
    println("Closed Solution: $x_star_closed_form")
    println("Closed Value: $(x_star_closed_form*d)")
end
println("Model retraining completed.")
