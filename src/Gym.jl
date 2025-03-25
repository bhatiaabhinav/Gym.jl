module Gym

using PythonCall
using MDPs
import MDPs: action_space, state_space, action_meaning, action_meanings, state, action, reward, reset!, step!, in_absorbing_state, truncated, visualize, info
using Random
using Colors

function translate_space(gym, pyspace)
    if pyisinstance(pyspace, gym.spaces.Box)
        lows = @pyconvert Array pyspace.low
        highs = @pyconvert Array pyspace.high
        T = lows |> eltype
        N = ndims(lows)
        return TensorSpace{T, N}(lows, highs)
    elseif pyisinstance(pyspace, gym.spaces.Discrete)
        n = @pyconvert Int pyspace.n
        return IntegerSpace(n)
    else
        error("$pyspace space not supported yet.")
    end
end


mutable struct GymEnv{S, A} <: AbstractMDP{S, A}
    const gym
    const pyenv
    const 𝕊
    const 𝔸
    const max_episode_steps::Real
    state::S
    action::A
    reward::Float64
    truncated::Bool
    terminated::Bool
    info::Dict{Symbol, Any}
    nsteps::Int
    is_old_api::Bool

    function GymEnv(gym, pyenv; is_old_api=false)
        if !is_old_api && !pyis(pyenv.render_mode, pybuiltins.None)
            @assert pyconvert(String, pyenv.render_mode) == "rgb_array" "Please set render_mode='rgb_array' in the gym environment."
            # println("Render mode is correctly set to rgb_array")
        end
        𝕊 = translate_space(gym, pyenv.observation_space)
        𝔸 = translate_space(gym, pyenv.action_space)
        S, A = eltype(𝕊), eltype(𝔸)
        max_episode_steps = pyhasattr(pyenv, "spec") && pyhasattr(pyenv.spec, "max_episode_steps") && !pyis(pyenv.spec.max_episode_steps, pybuiltins.None) ? pyconvert(Int, pyenv.spec.max_episode_steps) : Inf
        state = S == Int ? 1 : (!pyhasattr(pyenv, "state") || pyis(pyenv.state, pybuiltins.None)) ? zero(𝕊.lows) : pyconvert(S, pyenv.state)
        action = A == Int ? 1 : zero(𝔸.lows)
        
        return new{S, A}(gym, pyenv, 𝕊, 𝔸, max_episode_steps, state, action, 0.0, false, false, Dict{String, Any}(), 0, is_old_api)
    end
end

"""
    GymEnv(gym_env_name::String, args...; kwargs...)

Create a Gym environment with the given name. The arguments `args` and `kwargs` are passed to the `gym.make` function. The `render_mode` keyword argument is set to `"rgb_array"` by default.
"""
function GymEnv(gym, gym_env_name::String, args...; kwargs...)
    kwargs_dict = Dict{Symbol, Any}(kwargs...)
    kwargs_dict[:render_mode] = "rgb_array"
    pyenv = gym.make(gym_env_name, args...; kwargs_dict...)
    return GymEnv(gym, pyenv)
end

"""
    gym_wrap(env::GymEnv, pywrapper_class, args...; kwargs...)

Wrap a Gym environment with a Python wrapper class from `gym.wrappers`. The arguments `args` and `kwargs` are passed to the `pywrapper_class` constructor.
"""
function gym_wrap(env::GymEnv, pywrapper_class, args...; kwargs...)
    pyenv = pywrapper_class(env.pyenv, args...; kwargs...)
    return GymEnv(env.gym, pyenv)
end

@inline state_space(env::GymEnv) = env.𝕊
@inline action_space(env::GymEnv) = env.𝔸

function reset!(env::GymEnv{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    seed = rand(rng, 0:typemax(Int32))
    if !env.is_old_api
        obs, info = env.pyenv.reset(seed=seed)
    else
        obs = env.pyenv.reset(seed=seed)
    end
    obs = pyconvert(S, obs)
    if S isa AbstractArray
        copy!(env.state, obs)
    else
        env.state = obs
    end
    env.action = A == Int ? 1 : zero(env.𝔸.lows)
    env.reward = 0
    env.truncated = false
    env.terminated = false
    env.info = env.is_old_api ? Dict{Symbol, Any}() : pyconvert(Dict{Symbol, Any}, info)
    env.nsteps = 0
    nothing
end

function step!(env::GymEnv{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(env) "The action $a is not in the action space $(action_space(env))"
    np = env.gym.spaces.box.np
    env.action = a
    if in_absorbing_state(env)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        env.reward = 0.0
    elseif env.truncated
        @error "The environment has been truncated due to env.max_episode_steps=$(env.max_episode_steps). Gym will not step it further. This step! call will not do anything."
        env.reward = 0.0
    else
        if A == Int; a -=1; end
        env.nsteps += 1
        if !env.is_old_api
            obs, r, terminated, truncated, info = env.pyenv.step(a)
            if pyisinstance(terminated, np.bool_)
                terminated = pybool(terminated)
            end
            if pyisinstance(truncated, np.bool_)
                truncated = pybool(truncated)
            end
            env.terminated = pyconvert(Bool, terminated)
            env.truncated = pyconvert(Bool, truncated)
        else
            _a = np.asarray(a)
            obs, r, done, info = env.pyenv.step(_a)
            if pyisinstance(done, np.bool_)
                done = pybool(done)
            end
            done = pyconvert(Bool, done)
            if env.nsteps >= env.max_episode_steps
                if !done
                    @warn "Num steps has reached max_episode_steps but the environment is not `done`. Looks like the environment is not respecting max_episode_steps on its own." maxlog=1
                end
                env.truncated = true
                env.terminated = false  # assume that the last step is not terminal
            else
                env.truncated = false
                env.terminated = done
            end
        end
        obs = pyconvert(S, obs)
        if S isa AbstractArray
            copy!(env.state, obs)
        else
            env.state = obs
        end
        env.reward = pyconvert(Float64, r)
        env.info = pyconvert(Dict{Symbol, Any}, info)
    end
    nothing
end

@inline in_absorbing_state(env::GymEnv) = env.terminated
@inline truncated(env::GymEnv) = env.truncated
@inline info(env::GymEnv) = env.info

function visualize(env::GymEnv{S, A}, s::S; kwargs...) where {S, A}
    if S == Array{UInt8, 3}
        arr = reinterpret(Colors.N0f8, permutedims(state(env), (3, 1, 2)))
        img = reinterpret(reshape, RGB{eltype(arr)}, arr)
        return img
    end
end

function visualize(env::GymEnv; kwargs...)
    pyarr = env.pyenv.render()
    if PythonCall.pyis(pyarr, PythonCall.pybuiltins.None)
        @error "The environment is not rendering. Please try setting render_mode='rgb_array' in the gym environment. Trying visualize for the state" maxlog=1
        rgb_array = state(env)
    else
        rgb_array = pyconvert(Array, pyarr)
    end
    arr = reinterpret(Colors.N0f8, permutedims(rgb_array, (3, 1, 2)))
    img = reinterpret(reshape, RGB{eltype(arr)}, arr)
    return img
end

include("gymvec.jl")

export GymEnv, gym_wrap

end # module PGym
