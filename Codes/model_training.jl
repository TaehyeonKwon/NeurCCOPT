module ModelTraining

include("data_generation.jl")


using Flux, Distributions
using Flux.Data: DataLoader
using Random, LinearAlgebra, Statistics
using .DataGeneration: compute_quantile,create_dataset,split_dataset
using Plots


export prepare_train_dataset, train_NN, model_output_function_jl, model_validation

function prepare_train_dataset(X_train, Y_train ,params)
    X_train = vec(X_train)
    Y_train = vec(Y_train)
    return DataLoader((hcat(X_train...), hcat(Y_train...)), batchsize = params[:batch_size], shuffle = false)
end

function train_NN(train_dataset, params)
    Random.seed!(123)
    model = Chain(Dense(params[:d], 3,sigmoid), Dense(3, 1))
    optimizer = ADAM(params[:learning_rate])
    loss(x, y) = Flux.Losses.mse(model(x), y)
    model_params = Flux.params(model)
    for epoch in 1:params[:epochs]
        for batch in train_dataset
            Flux.train!(loss, model_params, [batch], optimizer)
        end
        if epoch % 5 == 0 
            current_loss = mean([loss(first(batch), last(batch)) for batch in train_dataset])
            # @info "Epoch: $epoch , Loss: $current_loss"
        end
    end


    # function custom_loss(x, y, model)
    #     pred = model(x)
    #     quantile_pred = vec(pred)
        
    #     loss_quantile = Flux.Losses.binarycrossentropy(quantile_pred, vec(y))
    #     pred_feasibility = cc_feasibility(quantile_pred)
    #     penalty = sum((1 .- pred_feasibility) .* abs.(quantile_pred .- y)) / length(pred_feasibility)
        
    #     return loss_quantile + 3*penalty
    # end
    
    # model_params = Flux.params(model)
    # for epoch in 1:params[:epochs]
    #     for batch in train_dataset
    #         x, y = batch
    #         gs = Flux.gradient(model_params) do
    #             custom_loss(x, y, model)
    #         end
    #         Flux.Optimise.update!(optimizer, model_params, gs)
    #     end

    #     if epoch % 5 == 0 
    #         current_loss = mean([custom_loss(batch[1], batch[2], model) for batch in train_dataset])
    #        #  @info "Epoch: $epoch , Loss: $current_loss"
    #     end
    # end

    return model
end





# function train_NN(train_dataset, params)
#     Random.seed!(123)
#     model = Chain(
#         Dense(params[:d] + 1, 3, relu),  # 입력 차원이 d + 1 (feasibility)
#         Dense(3, 1)                     # 출력은 하나의 quantile 값
#     )
#     optimizer = ADAM(params[:learning_rate])
    
#     function custom_loss(x, y, feasibility)
#         feasibility = convert(Matrix{Float64}, feasibility)  # feasibility를 Float64로 변환
#         input = hcat(x, feasibility)
#         pred = model(input)
#         quantile_pred = pred[:, 1]
        
#         loss_quantile = Flux.Losses.mse(quantile_pred, y)
#         # quantile_pred를 기반으로 feasibility를 판단
#         pred_feasibility = cc_feasibility(quantile_pred)
#         # 예측값이 feasible한 경우에 손실을 줄임
#         feasible_penalty = sum((quantile_pred .- y) .* pred_feasibility) / length(pred_feasibility)
        
#         return loss_quantile - feasible_penalty
#     end

#     model_params = Flux.params(model)
#     for epoch in 1:params[:epochs]
#         for batch in train_dataset
#             x, y, feas = batch
#             feas = convert(Matrix{Float64}, feas)  # batch 내의 feas를 Float64로 변환
#             gs = Flux.gradient(model_params) do
#                 custom_loss(x, y, feas)
#             end
#             Flux.Optimise.update!(optimizer, model_params, gs)
#         end
#         if epoch % 5 == 0 
#             current_loss = mean([custom_loss(batch[1], batch[2], convert(Matrix{Float64}, batch[3])) for batch in train_dataset])
#             @info "Epoch: $epoch , Loss: $current_loss"
#         end
#     end
#     return model
# end


function model_output_function_jl(x::Vector{Float64}, model)
    dense1_output = model[1].σ.(model[1].weight * x .+ model[1].bias)
    dense2_output = model[2].σ.(model[2].weight * dense1_output .+ model[2].bias)
    return dense2_output[1]
end

function model_validation(model, lower_bound, upper_bound, d)
    x = rand(Uniform(lower_bound, upper_bound), d)
    model_output = model(x)
    return isapprox(model_output[1], model_output_function_jl(x, model), atol = 1e-3)
end

end # module
