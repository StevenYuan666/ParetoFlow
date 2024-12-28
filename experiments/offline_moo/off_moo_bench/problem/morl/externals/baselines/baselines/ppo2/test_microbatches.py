from functools import partial

import gym
import numpy as np
import tensorflow as tf
from baselines.common.tf_util import make_session
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.microbatched_model import MicrobatchedModel
from baselines.ppo2.ppo2 import learn


def test_microbatches():
    def env_fn():
        env = gym.make("CartPole-v0")
        env.seed(0)
        return env

    learn_fn = partial(learn, network="mlp", nsteps=32, total_timesteps=32, seed=0)

    env_ref = DummyVecEnv([env_fn])
    sess_ref = make_session(make_default=True, graph=tf.Graph())
    learn_fn(env=env_ref)
    vars_ref = {v.name: sess_ref.run(v) for v in tf.trainable_variables()}

    env_test = DummyVecEnv([env_fn])
    sess_test = make_session(make_default=True, graph=tf.Graph())
    learn_fn(env=env_test, model_fn=partial(MicrobatchedModel, microbatch_size=2))
    # learn_fn(env=env_test)
    vars_test = {v.name: sess_test.run(v) for v in tf.trainable_variables()}

    for v in vars_ref:
        np.testing.assert_allclose(vars_ref[v], vars_test[v], atol=3e-3)


if __name__ == "__main__":
    test_microbatches()
