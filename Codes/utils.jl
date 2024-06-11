using Revise
using Distributions
using LinearAlgebra
using Random
using JuMP
using Ipopt
using Statistics
using Plots
using Flux
using Flux.Data: DataLoader

include("plot.jl")
include("data_generation.jl")
include("model_training.jl")
include("ccp_parameters.jl")

using .CCPParameters: setup_parameters
using .DataGeneration: compute_quantile,SAA,create_dataset,split_dataset, normalize_data, denormalize_data
using .ModelTraining: prepare_train_dataset, train_NN, model_output_function_jl, model_validation
using .ResultPlot: plot_quantile_predictions
