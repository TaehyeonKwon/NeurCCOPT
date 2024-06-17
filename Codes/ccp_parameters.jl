module CCPParameters

using Distributions

export setup_parameters

function setup_parameters(indicator)
    params = Dict()
    # problem parameter for hong.jl
    if indicator == 1
        # Params for Problem
        params[:d] = 5
        params[:epsilon] = 0.1 # original confidence level
        params[:alpha] = 0.15 # confidence level for quantile
        params[:delta] = 10^(-3) # confidence level for SAA
        params[:N_SAA] = log(1/params[:delta])/(2*(params[:alpha]-params[:epsilon])^2)
        # params[:N_SAA] = 10^5
        params[:m] = 5
        params[:beta] = (1 - params[:alpha])^(1 / params[:m])
        params[:lower_bound] = 0.0
        params[:upper_bound] = 10.0
        params[:case_type] = 0
        # params[:quantile_value] = params[:case_type] == 0 ? Distributions.quantile(Chisq(params[:d]), params[:beta]) : Distributions.quantile(Chisq(params[:d]), 1 - params[:alpha])
    end

    
    # problem parameter for credit_risk.jl
    if indicator ==2
        params[:d] = 50  # num of obilgor
        params[:epsilon] = 0.05
        params[:alpha] = 0.06 # confidence level for quantile
        params[:delta] = 10^(-3) # confidence level for sample
        params[:N_SAA] = log(1/params[:delta])/(2*(params[:alpha]-params[:epsilon])^2)
        # params[:N_SAA] = 10^6
        params[:w] = 250 
        params[:lower_bound] = -2.0
        params[:upper_bound] = 2.0
        params[:q] = collect(range(0, stop=10, length=params[:d]))
        params[:r] = collect(range(0.02, stop=0.12, length=params[:d]))
    end

    # Params for Framework
    params[:N] = 10^4 # sample for quantile
    params[:num_samples_x] = 100
    params[:seed] = 1
    params[:iterations] = 1000
    params[:K] = 30
    params[:theta] = 0.9 # should be adjusted dynamically

    # Don't change anymore 
    # Params for Neural Net Training
    params[:batch_size] = 1
    params[:hidden_layer] = round(Int, params[:d] / 3)
    params[:epochs] = 10^3
    params[:learning_rate] = 10^(-5)
    
    return params
end
end
