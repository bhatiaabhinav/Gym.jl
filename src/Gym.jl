module Gym

using PythonCall
using CondaPkg

const gym = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(gym, PythonCall.pyimport("gymnasium"))
end
IMPORTED_ROMS = false

"""
    ale_import_roms()

Import the roms for the Arcade Learning Environment. This is only necessary once.
"""
function ale_import_roms()
    global IMPORTED_ROMS
    if !IMPORTED_ROMS
        bin_dir = "$(CondaPkg.envdir())/bin"
        romspath = joinpath(@__DIR__, "..", "deps", "roms")
        run(`$bin_dir/ale-import-roms $romspath/`)
        IMPORTED_ROMS = true
    else
        println("Already imported roms")
    end
end


using MDPs
import MDPs: action_space, state_space, action_meaning, action_meanings, state, action, reward, reset!, step!, in_absorbing_state, truncated, visualize
using Random
using Colors

function translate_space(pyspace)
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
    const pyenv
    const 𝕊
    const 𝔸
    const max_episode_steps::Real
    state::S
    action::A
    reward::Float64
    truncated::Bool
    terminated::Bool

    function GymEnv(pyenv)
        @assert pyconvert(String, pyenv.render_mode) == "rgb_array" "Please set render_mode='rgb_array' in the gym environment."
        𝕊 = translate_space(pyenv.observation_space)
        𝔸 = translate_space(pyenv.action_space)
        S, A = eltype(𝕊), eltype(𝔸)
        max_episode_steps = pyhasattr(pyenv, "spec") && pyhasattr(pyenv.spec, "max_episode_steps") && !pyis(pyenv.spec.max_episode_steps, pybuiltins.None) ? pyconvert(Int, pyenv.spec.max_episode_steps) : Inf
        state = S == Int ? 1 : (!pyhasattr(pyenv, "state") || pyis(pyenv.state, pybuiltins.None)) ? zero(𝕊.lows) : pyconvert(S, pyenv.state)
        action = A == Int ? 1 : zero(𝔸.lows)
        
        return new{S, A}(pyenv, 𝕊, 𝔸, max_episode_steps, state, action, 0.0, false, false)
    end
end

"""
    GymEnv(gym_env_name::String, args...; kwargs...)

Create a Gym environment with the given name. The arguments `args` and `kwargs` are passed to the `gym.make` function. The `render_mode` keyword argument is set to `"rgb_array"` by default.
"""
function GymEnv(gym_env_name::String, args...; kwargs...)
    kwargs_dict = Dict{Symbol, Any}(kwargs...)
    kwargs_dict[:render_mode] = "rgb_array"
    pyenv = gym.make(gym_env_name, args...; kwargs_dict...)
    return GymEnv(pyenv)
end

"""
    gym_wrap(env::GymEnv, pywrapper_class, args...; kwargs...)

Wrap a Gym environment with a Python wrapper class from `gym.wrappers`. The arguments `args` and `kwargs` are passed to the `pywrapper_class` constructor.
"""
function gym_wrap(env::GymEnv, pywrapper_class, args...; kwargs...)
    pyenv = pywrapper_class(env.pyenv, args...; kwargs...)
    return GymEnv(pyenv)
end

@inline state_space(env::GymEnv) = env.𝕊
@inline action_space(env::GymEnv) = env.𝔸

function reset!(env::GymEnv{S, A}; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    seed = rand(rng, 1:typemax(Int))
    obs, info = env.pyenv.reset(seed=seed)
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
    nothing
end

function step!(env::GymEnv{S, A}, a::A; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing where {S, A}
    @assert a ∈ action_space(env) "The action $a is not in the action space $(action_space(env))"
    env.action = a
    if in_absorbing_state(env)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        env.reward = 0.0
    elseif env.truncated
        @error "The environment has been truncated due to env.max_episode_steps=$(env.max_episode_steps). Gym will not step it further. Please reduce the problem horizon so that this does not happen. This step! call will not do anything."
        env.reward = 0.0
    else
        if A == Int; a -=1; end
        obs, r, terminated, truncated, info = env.pyenv.step(a)
        obs = pyconvert(S, obs)
        if S isa AbstractArray
            copy!(env.state, obs)
        else
            env.state = obs
        end
        env.reward = pyconvert(Float64, r)
        env.terminated = pyconvert(Bool, terminated)
        env.truncated = pyconvert(Bool, truncated)
    end
    nothing
end

@inline in_absorbing_state(env::GymEnv) = env.terminated
@inline truncated(env::GymEnv) = env.truncated

function visualize(env::GymEnv{S, A}, s::S; kwargs...) where {S, A}
    if S == Array{UInt8, 3}
        arr = reinterpret(Colors.N0f8, permutedims(state(env), (3, 1, 2)))
        img = reinterpret(reshape, RGB{eltype(arr)}, arr)
        return img
    end
end

function visualize(env::GymEnv; kwargs...)
    rgb_array = pyconvert(Array, env.pyenv.render())
    arr = reinterpret(Colors.N0f8, permutedims(rgb_array, (3, 1, 2)))
    img = reinterpret(reshape, RGB{eltype(arr)}, arr)
    return img
end

export ale_import_roms, gym, GymEnv, gym_wrap

end # module PGym
