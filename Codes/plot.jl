module ResultPlot

export plot_quantile_predictions

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

end # module
