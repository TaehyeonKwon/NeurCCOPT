module DataGeneration

using Random, Distributions, LinearAlgebra

export create_dataset, generate_sample, sample_x, compute_quantile, normalize, split_dataset, cc_feasibility

function sample_x(lower_bound, upper_bound, num_samples_x, d)
    return [rand(Uniform(lower_bound, upper_bound), d) for _ in 1:num_samples_x]
end 

function generate_sample(seed, d, m, case_type)
    Random.seed!(seed)
    if case_type == 0
        return rand(Normal(0, 1), d, m)
    else 
        means = [j / d for j in 1:d]
        cov_matrix = fill(0.05, d, d)
        cov_matrix[diagind(cov_matrix)] .= 1.0
        return rand(MvNormal(means, cov_matrix), m)
    end
end

function compute_quantile(x, seed, N, d, m, alpha, case_type)
    results = Float64[]
    for i in 1:N
        sample_xi = generate_sample(seed + i, d, m, case_type)
        push!(results, cc_g(x, sample_xi))
    end
    sorted_results = sort(results, rev = true)
    index = ceil(Int, alpha * N)
    # error("stopping")
    return sorted_results[index]
end

function cc_g(x, sampled_xi)
    return maximum((dot(x.^2, sampled_xi[:, i].^2) - 100) for i in 1:size(sampled_xi, 2))
end

function create_dataset(params)
    X = sample_x(params[:lower_bound], params[:upper_bound], params[:num_samples_x], params[:d])
    Y = [compute_quantile(x, params[:seed], params[:N], params[:d], params[:m], params[:alpha], params[:case_type]) for x in X]
    feasibility = cc_feasibility(Y)
    return X, Y, feasibility
end

function normalize(data)
    min_val = minimum(data)
    max_val = maximum(data)
    # normalized_data = (data .- min_val) ./ (max_val - min_val)
    normalized_data = data
    return normalized_data, min_val, max_val
end

function split_dataset(X, Y, feasibility)
    train_set_end = floor(Int, length(X) * 0.8)
    X_train, X_test = X[1:train_set_end], X[train_set_end + 1:end]
    Y_train, Y_test = Y[1:train_set_end], Y[train_set_end + 1:end]
    feasibility_train, feasibility_test = feasibility[1:train_set_end], feasibility[train_set_end + 1:end]
    return X_train, X_test, Y_train, Y_test, feasibility_train, feasibility_test
end

function cc_feasibility(Y)
    return map(y -> y <= 0 ? 1.0 : 0.0, Y)
end

end
