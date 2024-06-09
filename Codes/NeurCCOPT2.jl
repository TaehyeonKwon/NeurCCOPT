# This code was implemented Hong et al.(2010) Section 5.1.1 problem
# Paper: https://pubsonline.informs.org/doi/pdf/10.1287/opre.1100.0910

using JuMP, Ipopt, Distributions, Flux
using Flux.Data: DataLoader
using Plots
using Statistics, Random
using LinearAlgebra


# 1. Given_Parameters
d = 5  # Degrees of freedom for ξ_i
alpha = 0.95  # Confidence level
w = 250
lower_bound = -2.0
upper_boudn = 2.0
q = collect(range(0, stop =10, length =d))
r = collect(range(0.02, stop=0.12, length=d))
n_samples = 1000


# 2. hyperparameters
N = 10^3 # number of scenario
num_samples_x = 1000  # initial generated number of x
epsilon = 100 # threshold for embedded nn constraint
seed = 1 # random seed for data generation
## 2.1 NN model
batch_size = 16
epochs = 30
learning_rate = 0.01
## 2.2 Framework Parameter
iterations = 1000
K = 30  # alternating sample size
theta = 0.9 # convexity parameter


function sample_x(lower_bound, upper_bound, num_samples_x)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 



function global_xi(seed)
    Random.seed!(seed)
    mu = collect(range(0.5, stop=2, length=d))
    sigma = fill(0.2, d, d) .+ (0.8 * Diagonal(ones(d)))
    u = collect(range(1, stop=4, length=d))
    M = MvNormal(mu, sqrt.(sigma) ./ 2)
    Xi = rand(M,d)
    return Xi
end


function cc_g(x,sample_xi)
   return (exp.(sample_xi) .- u).*x
end


function quantile(x, seed)
    results = Float64[] 
    for i in 1:N
        sample_xi = global_xi(seed + i)
        push!(results, cc_g(x, sample_xi))
    end
    sorted_results = sort(results; rev=true)
    index = ceil(Int, alpha * N)
    return sorted_results[index]
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
    dense1_output = model[1].σ.(model[1].weight * x .+ model[1].bias)
    dense2_output = model[2].σ.(model[2].weight * dense1_output .+ model[2].bias)
    return dense2_output[1]  
end

function model_validation(model)
    x=rand(Uniform(lower_bound, upper_bound), d)
    model_output = model(x)
    feasibility = isapprox(model_output[1],model_output_function_jl(x,model), atol=1e-5)
    return feasibility
end 




#### Case 2: obj function included to constraint ####

function solve_credit_risk_probelm(trained_nn)
    opt_model = Model(Ipopt.Optimizer)
    set_silent(opt_model)
    function f(x::Vector)
        x_val = copy(x)
        for i in 1:length(trained_nn)
            x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
        end
        return x_val[1]
    end   
    @variable(opt_model, lower_bound <= x[1:d] <= upper_bound) 
    @objective(opt_model, Min,-sum(q[i]*r[i]*x[i] for i in 1:d) / sum(q))
    @constraint(opt_model, sum(q[i] * x[i] for i in 1:d) == sum(q))
    for i in 1:d
        @constraint(opt_model, q[i] * x[i] <= 0.20*sum(q))
    end
    @operator(opt_model, new_const, d, (x...) -> f(collect(x)))
    @constraint(opt_model, new_const(x...) <= w)
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
        x_star_jump,optimal_value = solve_credit_risk_probelm(lower_bound, upper_bound, alpha,model)
        feasi_quantile = quantile(x_star_jump, seed)
        
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
println("Model retraining completed.")
