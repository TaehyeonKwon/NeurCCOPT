module DataGeneration

using Random, Distributions, LinearAlgebra

export compute_quantile,Sample_Average_Apporximation,create_dataset,split_dataset, normalize_data, denormalize_data





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


function Sample_Average_Apporximation(x, params, global_xi, cc_g)
    results = Float64[]
    for i in 1:params[:N_SAA]
        sample_xi = global_xi(params[:seed] + i, params)
        value = cc_g(x, sample_xi)
        push!(results, value <= 0 ? 1 : 0)
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


function q_hat(x, cc_g, sample_xi,params)
    sum = 0
    for i in 1:params[:N_SAA]
        sample_xi = global_xi(params[:seed] + i, params)
        sum += cc_g(x, sample_xi)<= 0 ? 1 : 0
    return sum / 1:params[:N_SAA]
end


function upper_confidence_bound(x, cc_g, sample_xi, alpha,params)
    N = params[:N_SAA]
    q_hat_value = q_hat(x, cc_g, sample_xi, params)
    z_alpha = quantile(Normal(0, 1), 1 - alpha)
    return q_hat_value + z_alpha * sqrt(q_hat_value * (1 - q_hat_value) / N)
end

function check_feasibility(x, cc_g, sample_xi, params)
    U_beta_N_prime = upper_confidence_bound(x, cc_g, sample_xi, params[:alpha],params)
    return U_beta_N_prime <= params[:epsilon]
end





end
