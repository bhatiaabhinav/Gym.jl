module Gym

using PyCall
const gym = PyNULL()
function __init__()
    copy!(gym, pyimport("gym"))
end

using MDPs
import MDPs: action_space, state_space, action_meaning, action_meanings, state, action, reward, reset!, step!, in_absorbing_state, truncated, visualize
using Random
using Colors

export GymEnv, gym

function translate_space(pyspace)
    if pyisinstance(pyspace, gym.spaces.Box)
        lows = pyspace.low
        highs = pyspace.high
        T = lows |> eltype
        N = ndims(lows)
        return TensorSpace{T, N}(lows, highs)
    elseif pyisinstance(pyspace, gym.spaces.Discrete)
        n = pyspace.n
        return IntegerSpace(n)
    else
        error("$pyspace space not supported yet.")
    end
end

mutable struct GymEnv{S, A} <: AbstractMDP{S, A}
    const pyenv
    const 𝕊
    const 𝔸
    const max_episode_steps::Real
    state::S
    action::A
    reward::Float64
    truncated::Bool
    terminated::Bool

    function GymEnv(gym_env_name, args...; kwargs...)
        kwargs_dict = Dict{Symbol, Any}(kwargs...)
        kwargs_dict[:render_mode] = "rgb_array"
        pyenv = gym.make(gym_env_name, args...; kwargs_dict...)
        𝕊 = translate_space(pyenv.observation_space)
        𝔸 = translate_space(pyenv.action_space)
        S, A = eltype(𝕊), eltype(𝔸)
        max_episode_steps = py"""hasattr($pyenv, "spec")""" && py"""hasattr($pyenv.spec, "max_episode_steps")""" && !isnothing(pyenv.spec.max_episode_steps) ? pyenv.spec.max_episode_steps : Inf
        if max_episode_steps < Inf
            @info "This is a finite horizon problem with max_episode_steps = $max_episode_steps. Gym will not step the environment after these many steps. Ensure that you set the horizon less than or equal to this when interacting with the environment."
        end
        state = S == Int ? 1 : ((py"""not hasattr($pyenv, "state")""" || isnothing(pyenv.state)) ? zero(𝕊.lows) : S(pyenv.state))
        action = A == Int ? 1 : zero(𝔸.lows)
        
        return new{S, A}(pyenv, 𝕊, 𝔸, max_episode_steps, state, action, 0.0, false, false)
    end
end

@inline state_space(env::GymEnv) = env.𝕊
@inline action_space(env::GymEnv) = env.𝔸
# @inline action_meaning

function reset!(env::GymEnv{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    seed = rand(rng, 1:typemax(Int))
    obs, info = env.pyenv.reset(seed=seed)
    @assert isa(obs, S)
    # if S == Int
        env.state = obs
    # else
    #     copy!(env.state, obs)
    # end
    env.action = A == Int ? 1 : zero(env.𝔸.lows)
    env.reward = 0
    env.truncated = false
    env.terminated = false
    nothing
end

function step!(env::GymEnv{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(env)
    if A == Int
        env.action = a
    else
        copy!(env.action, a)
    end
    if in_absorbing_state(env)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        env.reward = 0.0
    elseif env.truncated
        @error "The environment has been truncated due to env.max_episode_steps=$(env.max_episode_steps). Gym will not step it further. Please reduce the problem horizon so that this does not happen. This step! call will not do anything."
        env.reward = 0.0
    else
        if A == Int; a -=1; end
        obs, r, terminated, truncated, info = env.pyenv.step(a)
        # if S == Int
            env.state = obs
        # else
            # copy!(env.state, obs)
        # end
        env.reward = r
        env.terminated = terminated
        env.truncated = truncated
    end
    nothing
end

@inline in_absorbing_state(env::GymEnv) = env.terminated

@inline truncated(env::GymEnv) = env.truncated

function visualize(env::GymEnv{S, A}, s::S, args...; kwargs...) where {S, A}
    if S == Array{UInt8, 3}
        arr = reinterpret(Colors.N0f8, permutedims(state(env), (3, 1, 2)))
        img = reinterpret(reshape, RGB{eltype(arr)}, arr)
        return img
    end
end

function visualize(env::GymEnv, args...; kwargs...)
    rgb_array = env.pyenv.render()
    arr = reinterpret(Colors.N0f8, permutedims(rgb_array, (3, 1, 2)))
    img = reinterpret(reshape, RGB{eltype(arr)}, arr)
    return img
end

end # module Gym
