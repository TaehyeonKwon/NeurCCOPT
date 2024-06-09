module CCPParameters

using Distributions

export setup_parameters

function setup_parameters(indicator)
    params = Dict()
    # problem parameter for hong.jl
    if indicator == 1
        # Params for Problem
        params[:d] = 5
        params[:alpha] = 0.1
        params[:m] = 5
        params[:beta] = (1 - params[:alpha])^(1 / params[:m])
        params[:lower_bound] = 0.0
        params[:upper_bound] = 10.0
        params[:case_type] = 1
        # params[:quantile_value] = params[:case_type] == 0 ? Distributions.quantile(Chisq(params[:d]), params[:beta]) : Distributions.quantile(Chisq(params[:d]), 1 - params[:alpha])
    end

    # problem parameter for nonconvex.jl
    if indicator == 2
        # Params for Problem
        params[:d] = 5
        params[:alpha] = 0.1
        params[:m] = 5
        params[:beta] = (1 - params[:alpha])^(1 / params[:m])
        params[:lower_bound] = 0.0
        params[:upper_bound] = 99999999.0
        params[:case_type] = 0
        # params[:quantile_value] = params[:case_type] == 0 ? Distributions.quantile(Chisq(params[:d]), params[:beta]) : Distributions.quantile(Chisq(params[:d]), 1 - params[:alpha])
    end


    if indicator ==3
        params[:d] = 5  # num of obilgor
        params[:alpha] = 0.95  # CI for VaR 
        params[:lower_bound] = -2.0
        params[:upper_bound] = 2.0
    end



    # Params for Framework
    params[:N] = 10000
    params[:num_samples_x] = 100
    params[:epsilon] = 0
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
