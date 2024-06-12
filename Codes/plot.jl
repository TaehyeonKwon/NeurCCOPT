module ResultPlot

export plot_quantile_predictions,plot_feasible_frontier

using Plots
using JuMP, Ipopt, Distributions, Flux
using Flux.Data: DataLoader
using Plots
using Statistics, Random
using LinearAlgebra

function plot_quantile_predictions(X_test, Y_test, model, iteration)
    Y_pred = [model(x)[1] for x in X_test]
    Y_test_vec = vec(Y_test)
    min_val = minimum([Y_test_vec; Y_pred]) - 0.5  
    max_val = maximum([Y_test_vec; Y_pred]) + 0.5  
    scatterplot = scatter(Y_test_vec, Y_pred,
                          label = "Predicted vs Actual",
                          title = "Quantile Prediction Accuracy - Iteration $iteration",
                          xlabel = "Actual Quantile",
                          ylabel = "Predicted Quantile",
                          xlims = (min_val, max_val),
                          ylims = (min_val, max_val),
                          legend = :topleft,
                          size = (600, 600),
                          msize = 6,
                          mcolor = :blue,
                          grid = true)
    plot!(scatterplot, [min_val, max_val], [min_val, max_val],
          label = "Ideal Line y = x", line = (:red, :dash, 2), color = :red)
    savepath = "./plots"
    if !isdir(savepath)
        mkdir(savepath)
    end
    savefig(scatterplot, joinpath(savepath, "iteration_$iteration.png"))
end


function pareto_frontier(alpha::Vector{Float64}, obj_value::Vector{Float64})
    n = length(alpha)
    pareto_indices = []
    for i in 1:n
        dominated = false
        for j in 1:n
            if i != j && (alpha[j] <= alpha[i] && obj_value[j] <= obj_value[i]) && (alpha[j] < alpha[i] || obj_value[j] < obj_value[i])
                dominated = true
                break
            end
        end
        if !dominated
            push!(pareto_indices, i)
        end
    end
    return pareto_indices
end


function plot_feasible_frontier(alpha::Vector{Float64}, obj_value::Vector{Float64})
    pareto_indices = pareto_frontier(alpha, obj_value)
    sorted_indices = sort(pareto_indices, by=i -> alpha[i])
    scatter(alpha, obj_value, title="Feasible Frontier", xlabel="Alpha", ylabel="Objective Value", label="All Points", color=:blue)
    scatter!(alpha[sorted_indices], obj_value[sorted_indices], label="Pareto Frontier", color=:red, marker=:star5)
    plot!(alpha[sorted_indices], obj_value[sorted_indices], label="", color=:red, linewidth=2)
end

# Call this function
plot_feasible_frontier(alpha, obj_value)

end # module
