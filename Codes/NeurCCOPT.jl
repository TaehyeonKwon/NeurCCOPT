# This code was implemented Hong et al.(2010) Section 5.1.1 problem
# Paper: https://pubsonline.informs.org/doi/pdf/10.1287/opre.1100.0910

using JuMP, Ipopt, Distributions, Flux
using Flux.Data: DataLoader
using Plots
using Statistics, Random
using LinearAlgebra


# Parameters
d = 10  # Degrees of freedom for ξ_i
alpha = 0.05  # Confidence level
m = 10^4  # Number of scenarios
beta = (1 - alpha)^(1/m)
lower_bound = 0.0
upper_bound = 10.0
num_samples_x = 1000  # N
info = m
epsilon = 10^(-4)


# NN model
batch_size = 10
epochs = 30
learning_rate = 0.001

# Framework Parameter
iterations = 10
K = 10  # alternating sample size
theta = 0.2 # convexity parameter



function sample_x(lower_bound, upper_bound, num_samples_x)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 


function global_xi(info) 
    m = info
    xi_vector = rand(Normal(0,1), d, m)
    return xi_vector
end 


function cc_g(x, xi)
    constraint_value = 100 
    return dot(x.^2, xi.^2) - constraint_value
end



function log_SAA(x, num_scenarios)
    sampled_xi = global_xi(num_scenarios)
    count = sum(cc_g(x, sampled_xi[:,i]) > epsilon for i in 1:num_scenarios)   # ? 
    return log(count / num_scenarios)
end



function create_dataset(lower_bound, upper_bound, num_samples_x, info)
    X = sample_x(lower_bound, upper_bound, num_samples_x)
    Y = [log_SAA(x, info) for x in X]
    return X, Y
end



X, Y = create_dataset(lower_bound, upper_bound, num_samples_x, info)
println("Dataset generated.")

train_set_end = floor(Int, length(X)*0.8)
test_set_start = train_set_end+1

X_train, X_test = X[1:train_set_end], X[test_set_start:end]
Y_train, Y_test = Y[1:train_set_end], Y[test_set_start:end]

X_train = vec(X_train)
Y_train = vec(Y_train)
X_test = hcat(X_test...)
Y_test = hcat(Y_test...)

train_dataset = DataLoader((hcat(X_train...), hcat(Y_train...)), batchsize=batch_size, shuffle=false)


function train_NN(train_dataset)
    model = Chain(Dense(d, 3, sigmoid), Dense(3, 1))
    optimizer = ADAM(learning_rate)
    loss(x,y) = Flux.Losses.mse(model(x),y)
    params = Flux.params(model)  
    for epoch in 1:epochs
        for batch in train_dataset
            Flux.train!(loss, params, [batch], optimizer)
        end
        current_loss = mean([loss(first(batch), last(batch)) for batch in train_dataset])
        # @info "Epoch: $epoch , Loss:  $current_loss"
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


function solve_norm_opt_probelm(lower_bound, upper_bound, alpha,trained_nn)
    opt_model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(opt_model, "print_level", 0)
    function f(x::Vector)
        x_val = copy(x)
        for i in 1:length(trained_nn)
            x_val = (trained_nn[i].weight * x_val .+ trained_nn[i].bias)
        end
        return x_val[1]
    end    
    @variable(opt_model, x[1:d])
    for i in 1:d
        @constraint(opt_model, x[i] >= lower_bound)
        @constraint(opt_model, x[i] <= upper_bound)
    end
    @operator(opt_model, new_const, d, (x...) -> f(collect(x)))
    @constraint(opt_model, new_const(x...) <= log(alpha))
    @objective(opt_model, Min, -sum(x))
    optimize!(opt_model)
    return value.(x)
end

nn_model = train_NN(train_dataset)
println("Initial model training completed!")


function iterative_retraining(iterations, K, theta, model)
    for iteration in 1:iterations 
        @assert model_validation(model)
        x_star_jump = solve_norm_opt_probelm(lower_bound, upper_bound, alpha,model)
        feasi_probability = log_SAA(x_star_jump, info)
        if feasi_probability > log(alpha)
            println("Iteration $iteration: Infeasible solution found, x* = $x_star_jump")
            for k in 1:K
                bar_x = rand(Uniform(lower_bound, upper_bound), d)
                x_k = theta*x_star_jump + (1-theta)*bar_x
                push!(X_train, x_star_jump)
                push!(Y_train, log_SAA(x_k,info))
            end
            X_train_updated, Y_train_updated = hcat(X_train...), hcat(Y_train...)
            train_dataset = DataLoader((X_train_updated, Y_train_updated), batchsize=batch_size, shuffle=true)
            model = train_NN(train_dataset)
        else
            println("Iteration $iteration: Feasible solution found, x* = $x_star_jump. No retraining required.")
        end
    end
    return model
end

# Begin the iterative retraining process
retraining_nn_model = iterative_retraining(iterations, K, theta, nn_model)
println("Model retraining completed.")
