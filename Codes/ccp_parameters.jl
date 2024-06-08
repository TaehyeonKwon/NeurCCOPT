module CCPParameters

using Distributions

export setup_parameters

function setup_parameters()
    params = Dict()
    
    # Params for Problem
    params[:d] = 5
    params[:alpha] = 0.05
    params[:m] = 10
    params[:beta] = (1 - params[:alpha])^(1 / params[:m])
    params[:lower_bound] = 0.0
    params[:upper_bound] = 10.0
    params[:case_type] = 1
    # params[:quantile_value] = params[:case_type] == 0 ? Distributions.quantile(Chisq(params[:d]), params[:beta]) : Distributions.quantile(Chisq(params[:d]), 1 - params[:alpha])
    
    # Params for Framework
    params[:N] = 1000
    params[:num_samples_x] = 100
    params[:epsilon] = 10
    params[:seed] = 1
    params[:iterations] = 1000
    params[:K] = 30
    params[:theta] = 0.9

    # Params for Neural Net Training
    params[:batch_size] = 16
    params[:epochs] = 30
    params[:learning_rate] = 0.01
    
    return params
end
end
