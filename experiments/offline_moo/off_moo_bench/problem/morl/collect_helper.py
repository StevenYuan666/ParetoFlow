import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "externals/baselines"))
sys.path.append(os.path.join(base_dir, "externals/pytorch-a2c-ppo-acktr-gail"))

import pickle

import environments
import gym
import numpy as np
import torch
from a2c_ppo_acktr.model import Policy
from morl.sample import Sample

env_names = [
    "MO-Ant-v2",
    "MO-HalfCheetah-v2",
    "MO-Hopper-v2",
    "MO-Humanoid-v2",
    "MO-Swimmer-v2",
    "MO-Walker2d-v2",
]
env_names_and_infos = {
    "MO-Ant-v2": {
        "base_name": "Ant-v2",
        "max_task": 69,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-HalfCheetah-v2": {
        "base_name": "HalfCheetah-v2",
        "max_task": 233,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-Hopper-v2": {
        "base_name": "Hopper-v2",
        "max_task": 95,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-Humanoid-v2": {
        "base_name": "Humanoid-v2",
        "max_task": 354,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-Swimmer-v2": {
        "base_name": "Swimmer-v2",
        "max_task": 173,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-Walker2d-v2": {
        "base_name": "Walker2d-v2",
        "max_task": 364,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 2,
    },
    "MO-Hopper-v3": {
        "base_name": "Hopper-v3",
        "max_task": 2444,
        "num_processes": 1,
        "base_kwargs": {"layernorm": False},
        "obj_num": 3,
    },
}


def run_pgmorl(env_name):
    num_seeds = 1
    num_processes = 1
    command = f"python scripts/{env_name}.py --pgmorl --num_seeds {num_seeds} --num_processes {num_processes}"
    os.system(command)


def load_params(env_name, policy_path, data_dir):
    base_env = gym.make(env_name)
    env_info = env_names_and_infos[env_name]
    n_obj = env_info["obj_num"]
    actor_critic = Policy(
        action_space=base_env.action_space,
        obs_shape=base_env.observation_space.shape,
        base_kwargs=env_info["base_kwargs"],
        obj_num=n_obj,
    )
    actor_critic.eval()
    actor_critic.load_state_dict(torch.load(policy_path))

    params_shapes_path = os.path.join(data_dir, "params_shapes.pkl")
    if not os.path.exists(params_shapes_path):
        params_shapes = [
            p.cpu().detach().numpy().shape for p in actor_critic.parameters()
        ]
        with open(params_shapes_path, "wb+") as f:
            pickle.dump(params_shapes, f)

    flat_params = np.array([])
    for p in actor_critic.parameters():
        flat_param = p.detach().numpy().ravel()
        flat_params = np.concatenate((flat_params, flat_param))

    del actor_critic
    return flat_params


def get_results_dir(env_name):
    results_dir = os.path.join(
        base_dir, "results", f"{env_names_and_infos[env_name]['base_name']}", "pgmorl"
    )
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    return results_dir


def get_data_dir(env_name):
    data_dir = os.path.join(
        base_dir, "..", "..", "..", "data", f"{env_name.lower().replace('-', '_')}"
    )
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    return data_dir


def get_input_size(env_name):
    data_path = get_data_dir(env_name)
    params_shapes_path = os.path.join(data_path, "params_shapes.pkl")
    if os.path.exists(params_shapes_path):
        with open(params_shapes_path, "rb+") as f:
            params_shapes = pickle.load(f)
        from functools import reduce

        params_dims = reduce(lambda x, y: np.prod(x) + np.prod(y), params_shapes)
        return params_dims
    else:
        eval_env = gym.make(env_name)
        env_info = env_names_and_infos[env_name]

        n_obj = env_info["obj_num"]
        policy = Policy(
            action_space=eval_env.action_space,
            obs_shape=eval_env.observation_space.shape,
            base_kwargs=env_info["base_kwargs"],
            obj_num=n_obj,
        )
        problem_dims = 0
        for p in policy.parameters():
            problem_dims += np.prod(p.cpu().detach().numpy())
        return problem_dims
