include("utils.jl")
include("optimization.jl")
include("problems/hong.jl")

using .Optimization: iterative_retraining
using .Hong: AbstractCCP, NormCCP, NeuralModel

function run(P::AbstractCCP; params)
    X, Y, Feasibility = create_dataset(params)
    Y_normalized, Y_min, Y_max = normalize(Y)
    params[:Y_min] = Y_min
    params[:Y_max] = Y_max
    X_train, X_test, Y_train, Y_test, feasibility_train, feasibility_test = split_dataset(X, Y_normalized, Feasibility)

    train_dataset = prepare_train_dataset(X_train, Y_train, feasibility_train, params)
    println("Dataset Generated for Training!")

    nn_model = train_NN(train_dataset, params)
    println("Initial Model trained!")
    retrained_model = iterative_retraining(P, nn_model, X_train, Y_train, params)
    println("Model retraining completed.")
end

params = setup_parameters()
P = NormCCP()

run(P; params)

