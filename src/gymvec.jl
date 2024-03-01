using MDPs

export GymVecEnv

mutable struct GymVecEnv{S, A} <: AbstractVecEnv{S, A}
    const pyenv
    const 𝕊
    const 𝔸
    const num_envs::Int
    states::Vector{S}
    states_after_autoreset::Vector{Union{S, Nothing}}
    infos_after_autoreset::Vector{Union{Dict{Symbol, Any}, Nothing}}
    actions::Vector{A}
    rewards::Vector{Float64}
    truncateds::Vector{Bool}
    terminateds::Vector{Bool}
    infos::Vector{Dict{Symbol, Any}}
    function GymVecEnv(pyenv)
        𝕊 = translate_space(pyenv.single_observation_space)
        𝔸 = translate_space(pyenv.single_action_space)
        S, A = eltype(𝕊), eltype(𝔸)
        num_envs = pyconvert(Int, pyenv.num_envs)
        state = S == Int ? 1 : (!pyhasattr(pyenv, "state") || pyis(pyenv.state, pybuiltins.None)) ? zero(𝕊.lows) : pyconvert(S, pyenv.state)
        action = A == Int ? 1 : zero(𝔸.lows)
        states = fill(state, num_envs)
        states_after_autoreset = fill(nothing, num_envs)
        infos_after_autoreset = fill(nothing, num_envs)
        actions = fill(action, num_envs)
        truncateds = falses(num_envs)
        terminateds = falses(num_envs)
        info = Dict{Symbol, Any}()
        infos = fill(info, num_envs)
        rewards = zeros(Float64, num_envs)
        return new{S, A}(pyenv, 𝕊, 𝔸, num_envs, states, states_after_autoreset, infos_after_autoreset, actions, rewards, truncateds, terminateds, infos)
    end
end


"""
    GymVecEnv(gym_env_name::String, num_envs::Int, args...; kwargs...)

Create a GymVec environment with the given name and the number of parallel environment instances. The arguments `args` and `kwargs` are passed to the `gym.vector.make` function.
"""
function GymVecEnv(gym_env_name::String, num_envs::Int, args...; kwargs...)
    pyenv = gym.vector.make(gym_env_name, num_envs, args...; kwargs...)
    return GymVecEnv(pyenv)
end

function MDPs.get_envs(v::GymVecEnv{S, A}) where {S, A}
    error("Cannot retrieve individual environments from a GymVecEnv.")
end

@inline function MDPs.num_envs(v::GymVecEnv{S, A}) where {S, A}
    v.num_envs
end

@inline state_space(v::GymVecEnv) = v.𝕊
@inline action_space(v::GymVecEnv) = v.𝔸

function action_meaning(v::GymVecEnv{S, A}, a::A)::String where {S, A}
    return "Action $a"
end
function action_meanings(v::GymVecEnv{S, Int})::Vector{String}where {S}
    return map(a -> action_meaning(v, a), action_space(v))
end

function state(v::GymVecEnv{Array{Tₛ, N}, A})::AbstractArray{Tₛ} where {Tₛ, N, A}
    if N == 1
        return reduce(hcat, v.states)
    else
        return reduce((x, y) -> cat(x, y, dims=N+1), v.states)
    end
end

function state(v::GymVecEnv{S, A})::Vector{S} where {S, A}
    return v.states
end

function action(v::GymVecEnv{S, Array{Tₐ, N}})::AbstractArray{Tₐ} where {S, Tₐ, N}
    if N == 1
        return reduce(hcat, v.actions)
    else
        return reduce((x, y) -> cat(x, y, dims=N+1), v.actions)
    end
end

function action(v::GymVecEnv{S, A})::Vector{A} where {S, A}
    return v.actions
end

function reward(v::GymVecEnv{S, A})::Vector{Float64} where {S, A}
    return v.rewards
end

function in_absorbing_state(v::GymVecEnv{S, A})::Vector{Bool} where {S, A}
    return v.terminateds
end

function truncated(v::GymVecEnv{S, A})::Vector{Bool} where {S, A}
    return v.truncateds
end

function info(v::GymVecEnv{S, A})::Vector{Dict{Symbol, Any}} where {S, A}
    return v.infos
end


function reset!(v::GymVecEnv{S, A}, reset_all::Bool=true; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    if !reset_all
        for i in 1:v.num_envs
            if v.states_after_autoreset[i] !== nothing
                if S isa AbstractArray
                    copy!(v.states[i], v.states_after_autoreset[i])
                else
                    v.states[i] = v.states_after_autoreset[i]
                end
                v.actions[i] = A == Int ? 1 : zero(v.𝔸.lows)
                v.rewards[i] = 0
                v.truncateds[i] = false
                v.terminateds[i] = false
                v.infos[i] = v.infos_after_autoreset[i]
                v.states_after_autoreset[i] = nothing
                v.infos_after_autoreset[i] = nothing
            end
        end
    else
        seeds = rand(rng, 1:typemax(Int), v.num_envs)
        obs, info = v.pyenv.reset(seed=seeds)
        v.infos = vecinfos_to_vector_of_infos(info, v.num_envs)
        for i in 1:v.num_envs
            obsᵢ = pyconvert(S, obs[i-1])
            if S isa AbstractArray
                copy!(v.states[i], obsᵢ)
            else
                v.states[i] = obsᵢ
            end
            v.actions[i] = A == Int ? 1 : zero(v.𝔸.lows)
            v.rewards[i] = 0
            v.truncateds[i] = false
            v.terminateds[i] = false
        end
    end
    nothing
end

function step!(v::GymVecEnv{S, A}, action; rng::AbstractRNG=Random.GLOBAL_RNG) where {S, A}
    if any(v.states_after_autoreset .!== nothing)
        error("Cannot step! a GymVecEnv that has an environment in terminated or truncated state. Please call reset! first (with reset_all=false if you want to reset only the environments that are in terminated or truncated state).")
    end
    
    # save action to actions:
    if A == Int
        v.actions .= action
    else
        for i in 1:v.num_envs
            copy!(v.actions[i], selectdim(action, ndims(action), i))
        end
    end

    if A == Int
        action .-= 1
    else
        action = transpose(action) |> copy # since python is row-major
        action = np.asarray(action)
    end
    obs, r, terminated, truncated, info = v.pyenv.step(action)
    v.rewards .= pyconvert(Vector{Float64}, r)
    v.terminateds .= pyconvert(Vector{Bool}, terminated)
    v.truncateds .= pyconvert(Vector{Bool}, truncated)
    if any(v.terminateds) || any(v.truncateds)
        final_info = info["final_info"]
        final_observation = info["final_observation"]
        pydelitem(info, "final_info")
        pydelitem(info, "final_observation")
    end
    infos = vecinfos_to_vector_of_infos(info, v.num_envs)
    for i in 1:v.num_envs
        obsᵢ = pyconvert(S, obs[i-1])
        if !v.terminateds[i] && !v.truncateds[i]
            if S isa AbstractArray
                copy!(v.states[i], obsᵢ)
            else
                v.states[i] = obsᵢ
            end
            v.infos[i] = infos[i]
        else
            final_obs_i = pyconvert(S, final_observation[i-1])
            if S isa AbstractArray
                copy!(v.states[i], final_obs_i)
            else
                v.states[i] = final_obs_i
            end
            v.infos[i] = pyconvert(Dict{Symbol, Any}, final_info[i-1])
            v.states_after_autoreset[i] = obsᵢ
            v.infos_after_autoreset[i] = infos[i]
        end
    end
    nothing
end

function vecinfos_to_vector_of_infos(gym_infos, num_envs)
    gym_infos = pyconvert(Dict{Symbol, Any}, gym_infos)
    infos = fill(Dict{Symbol, Any}(), num_envs)
    for (k, v) in gym_infos
        if startswith(string(k), "_")
            continue
        end
        for i in 1:num_envs
            infos[i][k] = v[i]
        end
    end
    return infos
end