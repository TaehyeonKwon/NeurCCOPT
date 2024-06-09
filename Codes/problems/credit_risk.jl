module Credit

include("../utils.jl")

using Convex, SCS, Random
using LinearAlgebra

export CreditProblem, sample_x, global_xi, cc_g, neurconst, credit_problem



function sample_x(params)
    return [rand(Uniform(params[:lower_bound], params[:upper_bound]), params[:d]) for _ in 1:params[:num_samples_x]]
end 


function global_xi(seed, params)
    Random.seed!(seed)
    μ = collect(range(0.5, stop=2, length=params[:d]))
    σ = 0.5 .* μ
    corr = 0.2
    Σ = Diagonal(σ .^ 2) + fill(corr, params[:d], params[:d])

    ε = 1e-5
    Σ += ε * I

    η = MvNormal(μ, Σ)
    u = collect(range(1, stop=4, length=params[:d]))
    ξ = exp.(rand(η) .- u)
    return ξ
end


function cc_g(x,sample_xi)
   return dot(sample_xi, x)-250
end


function neurconst(x::Vector, trained_nn)
    x_val = copy(x)
    for i in 1:length(trained_nn)
        x_val = trained_nn[i].σ.(trained_nn[i].weight * x_val .+ trained_nn[i].bias)
    end
    return x_val[1]
end 

struct CreditProblem
    trained_nn::Chain
    params::Dict{Symbol, Any}

    function CreditProblem(trained_nn,params)
        new(trained_nn, params)
    end
end


function credit_problem(problem::CreditProblem)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, problem.params[:lower_bound] <= x[1:problem.params[:d]] <= problem.params[:upper_bound]) 
    @objective(model, Min, (-sum(problem.params[:q][i]*problem.params[:r][i]*x[i] for i in 1:problem.params[:d]) / sum(problem.params[:q])))
    @constraint(model, sum(problem.params[:q][i] * x[i] for i in 1:problem.params[:d]) == sum(problem.params[:q]))
    for i in 1:problem.params[:d]
        @constraint(model, problem.params[:q][i] * x[i] <= 0.20*sum(problem.params[:q]))
    end
    @operator(model, new_const, problem.params[:d], (x...) -> neurconst(collect(x),problem.trained_nn))
    @constraint(model, new_const(x...) <= 0)
    optimize!(model)
    if !is_solved_and_feasible(model; allow_almost = true)
        # @show(termination_status(opt_model))
        # @warn("Unable to find a feasible and/or optimal solution of the embedded model")
    end
    return value.(x), -objective_value(model)
end



end



