module ModelTraining

include("data_generation.jl")


using Flux, Distributions
using Flux.Data: DataLoader
using Random, LinearAlgebra, Statistics
using .DataGeneration: compute_quantile,Sample_Average_Apporximation,create_dataset,split_dataset
using Plots


export prepare_train_dataset, train_NN, model_output_function_jl, model_validation

function prepare_train_dataset(X_train, Y_train ,params)
    X_train = vec(X_train)
    Y_train = vec(Y_train)
    return DataLoader((hcat(X_train...), hcat(Y_train...)), batchsize = params[:batch_size], shuffle = false)
end

function train_NN(train_dataset, params)
    Random.seed!(123)
    model = Chain(Dense(params[:d], params[:hidden_layer],sigmoid),
            Dense(params[:hidden_layer],1,relu))
    optimizer = ADAM(params[:learning_rate])
    loss(x, y) = Flux.Losses.mse(model(x), y)
    model_params = Flux.params(model)
    for epoch in 1:params[:epochs]
        for batch in train_dataset
            Flux.train!(loss, model_params, [batch], optimizer)
        end
        if epoch % 5 == 0 
            current_loss = mean([loss(first(batch), last(batch)) for batch in train_dataset])
            @info "Epoch: $epoch , Loss: $current_loss"
        end
    end
    return model
end


function model_output_function_jl(x::Vector{Float64}, model)
    x_val = copy(x)
    for i in 1:length(model)
        x_val = model[i].σ.(model[i].weight*x_val .+ model[i].bias)
    end
    return x_val[1]
end


function model_validation(model, lower_bound, upper_bound, d)
    x = rand(Uniform(lower_bound, upper_bound), d)
    model_output = model(x)
    return isapprox(model_output[1], model_output_function_jl(x, model), atol = 1e-3)
end

end # module
