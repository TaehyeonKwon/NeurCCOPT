module DataGeneration

using Random, Distributions, LinearAlgebra

export compute_quantile,SAA,create_dataset,split_dataset

function compute_quantile(x, params, global_xi, cc_g)
    results = Float64[]
    for i in 1:params[:N]
        sample_xi = global_xi(params[:seed] + i, params)
        push!(results, cc_g(x, sample_xi))
    end
    sorted_results = sort(results, rev = true)
    index = ceil(Int, params[:alpha] * params[:N])
    return sorted_results[index]
end


function SAA(x, params, global_xi, cc_g)
    results = Float64[]
    for i in 1:params[:N]
        sample_xi = global_xi(params[:seed] + i, params)
        value = cc_g(x, sample_xi)
        push!(results, value <= 0 ? value : 0)
    end
    return mean(results)
end




function create_dataset(params, sample_x, global_xi, cc_g)
    X = sample_x(params)
    Y = [compute_quantile(x, params, global_xi, cc_g) for x in X]
    return X, Y
end

function split_dataset(X, Y)
    train_set_end = floor(Int, length(X) * 0.8)
    X_train, X_test = X[1:train_set_end], X[train_set_end + 1:end]
    Y_train, Y_test = Y[1:train_set_end], Y[train_set_end + 1:end]
    return X_train, X_test, Y_train, Y_test
end


function normalize_data(Y)
    Y_min = minimum(Y)
    Y_max = maximum(Y)
    Y_normalized = (Y .- Y_min) ./ (Y_max .- Y_min)

    return Y_normalized, Y_min, Y_max
end


function denormalize_data(Y_normalized, Y_min, Y_max)
    Y_denormalized = Y_normalized .* (Y_max .- Y_min) .+ Y_min
    return Y_denormalized
end


end
