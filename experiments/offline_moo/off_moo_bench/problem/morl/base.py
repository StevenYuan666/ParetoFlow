import os
import pickle
import sys

import gym
import numpy as np
import torch

from offline_moo.off_moo_bench.problem.base import BaseProblem
from offline_moo.off_moo_bench.problem.morl.collect_helper import *

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "externals/baselines"))
sys.path.append(os.path.join(base_dir, "externals/pytorch-a2c-ppo-acktr-gail"))

import random

import environments
from a2c_ppo_acktr.model import Policy
from morl.sample import Sample


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class MORLProblem(BaseProblem):
    def __init__(
        self,
        name,
        n_obj,
        env_name,
        problem_type="continuous",
        nadir_point=None,
        ideal_point=None,
    ):
        assert env_name in env_names
        self.env_name = env_name
        self.env_info = env_names_and_infos[env_name]
        self.data_dir = get_data_dir(env_name)
        self.x_file = os.path.join(self.data_dir, f"{env_name}-x.npy")
        self.y_file = os.path.join(self.data_dir, f"{env_name}-y.npy")
        n_dim = get_input_size(env_name)
        super().__init__(
            name,
            problem_type,
            n_obj,
            n_dim,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )

    def generate_x(self, size):
        if os.path.exists(self.x_file):
            os.system(f"rm {self.x_file}")
        if os.path.exists(self.y_file):
            os.system(f"rm {self.y_file}")

        env_info = env_names_and_infos[self.env_name]
        results_dir = get_results_dir(self.env_name)
        n_obj = env_info["obj_num"]

        for seed_dir in os.listdir(results_dir):
            if os.path.isdir(os.path.join(results_dir, seed_dir)):
                final_results_dir = os.path.join(results_dir, seed_dir, "final")
                if not os.path.exists(final_results_dir):
                    continue

                i = 0
                while True:
                    policy_path = os.path.join(final_results_dir, f"EP_policy_{i}.pt")
                    if not os.path.exists(policy_path):
                        break

                    if os.path.exists(self.x_file):
                        x = np.load(self.x_file)
                    else:
                        x = None

                    params = load_params(self.env_name, policy_path, self.data_dir)
                    if x is None:
                        x = params.reshape((1, -1))
                    else:
                        x = np.concatenate((x, params.reshape(1, -1)), axis=0)

                    print(x.shape)
                    np.save(file=self.x_file, arr=x)
                    del x
                    i += 1

        x_ret = np.load(self.x_file)
        np.random.shuffle(x_ret)
        assert x_ret.shape[0] >= size, "Param data_size is too large."
        rows_to_remove = np.random.choice(
            np.arange(0, x_ret.shape[0]), x_ret.shape[0] - size, replace=False
        )
        x_ret = np.delete(x_ret, rows_to_remove, axis=0)
        print(x_ret.shape)
        return x_ret

    def evaluate(self, x, eval_seed=2023, *args, **kwargs):
        set_seed(eval_seed)

        env_name = self.env_name
        env_info = self.env_info
        eval_env = gym.make(env_name)
        n_obj = env_info["obj_num"]

        params_shapes_path = os.path.join(self.data_dir, "params_shapes.pkl")
        assert os.path.exists(
            params_shapes_path
        ), f"Params_shapes path {params_shapes_path} not found."
        with open(params_shapes_path, "rb+") as f:
            params_shapes = pickle.load(f)

        y = None

        for x0 in x:
            policy = Policy(
                action_space=eval_env.action_space,
                obs_shape=eval_env.observation_space.shape,
                base_kwargs=env_info["base_kwargs"],
                obj_num=n_obj,
            )
            policy.eval()
            policy.to("cpu").double()

            start = 0
            model_state_dict = policy.state_dict()

            for name, shape in zip(model_state_dict, params_shapes):
                end = start + np.prod(shape)
                param = x0[start:end].reshape(shape)
                model_state_dict[name] = torch.from_numpy(param)
                start = end

            policy.load_state_dict(model_state_dict)

            eval_seed = random.randint(1, 1000000)

            with torch.no_grad():
                eval_env.seed(eval_seed)
                obs = eval_env.reset()
                done = False
                ep_raw_reward = np.zeros(n_obj)
                while not done:

                    # reload normalizing value used when training behavioral policy
                    # ob_norm = np.clip((obs - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                    ob_norm = obs
                    action = policy.act(
                        torch.Tensor(ob_norm).double().unsqueeze(0),
                        None,
                        None,
                        deterministic=True,
                    )[1]

                    if self.env_name in ["MO-Hopper-v2", "MO-Hopper-v3"]:
                        action = np.clip(action, [-2, -2, -4], [2, 2, 4])
                    else:
                        action = np.clip(action, -1, 1)

                    next_obs, _, done, info = eval_env.step(action)
                    raw_reward = info["obj"]
                    obs = next_obs
                    ep_raw_reward += raw_reward

                y = (
                    ep_raw_reward.reshape(1, -1)
                    if y is None
                    else np.concatenate((y, ep_raw_reward.reshape(1, -1)), axis=0)
                )
                # if not os.path.exists(self.y_file):
                #     np.save(file=self.y_file, arr=ep_raw_reward.reshape(1,-1))
                # else:
                #     y = np.load(self.y_file)
                #     y = np.concatenate((y, ep_raw_reward.reshape(1,-1)), axis=0)
                #     np.save(file=self.y_file, arr=y)
                #     print(y.shape)
                #     del y

            del policy
        eval_env.close()
        # y_ret = np.load(self.y_file)
        return -y

    def get_nadir_point(self):
        return self.nadir_point

    def get_ideal_point(self):
        return self.ideal_point
