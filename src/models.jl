using Combinatorics
using StatsBase

function get_strategies(N_s::Int, N_p::Int, P::Int, S::Int; normal_condition::Bool=false, seed=nothing, s_strategies::Array{Int, 3}=Array{Int}(P, S+1, N_s), p_strategies::Array{Int, 3}=Array{Int}(P, 1, N_p))
    if seed != nothing
        srand(seed)
    end
    # initialize strategies of speculators

    for i in 1:N_s
        for j in 1:S
            if normal_condition
                s_strategies[:, j, i] = ones(Int, P)
                s_strategies[sample(1:P, div(P, 2), replace=false), j, i] = -1
            else
                s_strategies[:, j, i] = rand([-1, 1], P)
            end
        end
        s_strategies[:, S+1, i] = zeros(Int, P)
    end

    # initialize strategies of producers

    for i in 1:N_p
        p_strategies[:, 1, i] = rand([-1, 1], P)
    end

    return s_strategies, p_strategies
end

mutable struct DMGHLS
    N::Int
    P::Int
    S::Int
    lambda::Float64
    P_irr::Float64
    beta::Vector{Float64}
    price::Float64
    strategies::Array{Int, 3}
    U::Array{Float64, 2}

    function DMGHLS(N::Int, P::Int, S::Int, lambda::Float64, P_irr::Float64, U::Matrix{Float64}, beta::Vector{Float64}=Vector{Float64}(N); price::Float64=1.0, seed=nothing)
        strategies, _ = get_strategies(N, 0, P, S, seed=seed, normal_condition=true)
        return new(N, P, S, lambda, P_irr, beta, price, strategies, U)
    end
end

function initialize_DMGHLS(dmghls::DMGHLS; seed=nothing)
    strategies, _ = get_strategies(N, 0, P, S, seed=seed, s_strategies=dmghls.strategies, normal_condition=true)
    return dmghls
end

mutable struct GCMG
    N_s::Int
    N_p::Int
    P::Int
    S::Int
    lambda::Float64
    epsilon::Float64
    price::Float64
    s_strategies::Array{Int, 3}
    p_strategies::Array{Int, 3}
    U_s::Array{Float64, 2}

    function GCMG(N_s::Int, N_p::Int, P::Int, S::Int, lambda::Float64, U_s::Matrix{Float64}; epsilon::Float64=0.0, price::Float64=1.0, seed=nothing)
        s_strategies, p_strategies = get_strategies(N_s, N_p, P, S, seed=seed)
        return new(N_s, N_p, P, S, lambda, epsilon, price, s_strategies, p_strategies, U_s)
    end
end

function initialize_GCMG(gcmg::GCMG; seed=nothing) # memoery saver
    s_strategies, p_strategies = get_strategies(N_s, N_p, P, S, seed=seed, s_strategies=gcmg.s_strategies, p_strategies=gcmg.p_strategies)
    return gcmg
end

function simulate!(gcmg::GCMG, loops::Int; seed=nothing)
    if seed != nothing
        srand(seed)
    end
    # initialization
    s_strategies_taken = Array{Int}(gcmg.N_s)
    prices = Vector{Float64}(loops+1)
    prices[1] = gcmg.price
    Qs = Vector{Float64}(loops)
    actives = Vector{Int}(loops)
    fill!(actives, gcmg.N_p)

    # each step
    for l in 1:loops
        mu = rand(1:gcmg.P)
        Q = 0
        for i in 1:gcmg.N_s
            best_strategy = indmax(gcmg.U_s[:, i]) # choose the best scored strategy
            Q += gcmg.s_strategies[mu, best_strategy, i]
            s_strategies_taken[i] = best_strategy
            if best_strategy != (gcmg.S + 1)
                actives[l] += 1
            end
        end

        for i in 1:gcmg.N_p
            Q += gcmg.p_strategies[mu, 1, i]
        end

        r = Q / gcmg.lambda
        Qs[l] = Q

        for i in 1:gcmg.N_s
            for strategy in 1:gcmg.S
                gcmg.U_s[strategy, i] += -gcmg.s_strategies[mu, strategy, i] * Q
            end
            gcmg.U_s[gcmg.S+1, i] +=  gcmg.epsilon
        end

        gcmg.price *= exp(r)
        prices[l+1] = gcmg.price
    end

    return prices, Qs, actives
end


function simulate!(dmghls::DMGHLS, loops::Int; seed=nothing)
    if seed != nothing
        srand(seed)
    end
    # initialization
    strategies_taken_1 = Array{Int}(dmghls.N)
    strategies_taken_2 = zeros(Int, dmghls.N)
    prices = Vector{Float64}(loops+1)
    inactives = zeros(Int, loops)
    prices[1] = dmghls.price

    # each step
    for l in 1:loops
        strategies_taken_1 = strategies_taken_2
        mu = rand(1:dmghls.P)
        Q = 0
        r = 0
        for i in 1:dmghls.N
            if rand() < dmghls.P_irr
                Q += rand() < 0.5 ? 1 : -1
                strategies_taken_2[i] = 0
            else
                best_strategy = indmax(dmghls.U[:, i]) # choose the best scored strategy
                Q += dmghls.strategies[mu, best_strategy, i]
                strategies_taken_2[i] = best_strategy
                if best_strategy == dmghls.S+1
                    inactives[l] += 1
                end
            end
        end

        r = Q / dmghls.lambda

        for i in 1:dmghls.N
            strategy = strategies_taken_1[i]
            if strategy != 0
                #dmghls.U[strategy, i] += -dmghls.strategies[mu, strategy, i] * Q + (strategy == dmghls.S + 1 ? dmghls.epsilon : 0)
                dmghls.U[strategy, i] += beta[i] * (dmghls.strategies[mu, strategy, i] * r - dmghls.U[strategy, i])
            end
        end

        dmghls.price *= exp(r)
        prices[l+1] = dmghls.price
    end

    actives = dmghls.N .- inactives

    return prices, actives
end
