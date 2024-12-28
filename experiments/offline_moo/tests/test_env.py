import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import numpy as np
from utils import read_data

from offline_moo.off_moo_bench.problem import get_problem

# from evoxbench.database.init import config
# import os
# base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
#                          'm2bo_bench', 'problem', 'mo_nas')
# config(os.path.join(base_path, 'database'), os.path.join(base_path, 'data'))

env_names = [
    "zdt1",
    "re21",
    "molecule",
    "zinc",
    "regex",
    "rfp",
    "mo_hopper_v2",
    "c10mop1",
    "nb201_test",
]

for env_name in env_names:
    print(env_name)

    # try:
    problem = get_problem(env_name)
    x, y, _ = read_data(env_name, return_rank=False)
    print(problem.n_obj, problem.n_dim)
    print(type(problem))

    print(np.max(y, axis=0), np.min(y, axis=0))
    print("Nadir Point:", problem.get_nadir_point())
    print(x.shape)
    print(y.shape)

    print("x[100] =", x[100])
    print("f(x[100]) =", problem.evaluate(x[100].reshape(1, -1)), "y[100] =", y[100])
    print(len(np.where(np.isinf(y))[0]))
    print(len(np.where(np.logical_or(np.isinf(y), np.isnan(y)))[0]))

    # except:
    #     if env_name == 'nb201_test':
    #         problem = get_problem(env_name)
    #         x, y, _ = read_data(env_name, return_rank=False)
    #         print(problem.n_obj, problem.n_dim)
    #         print(type(problem))

    #         print(np.max(y, axis=0), np.min(y, axis=0))
    #         print('Nadir Point:', problem.get_nadir_point())
    #         print(x.shape)
    #         print(y.shape)

    #         print('x[100] =', x[100])
    #         print('f(x[100]) =', problem.evaluate(x[100].reshape(1, -1)), 'y[100] =', y[100])
    #         print(len(np.where(np.isinf(y))[0]))
    #         print(len(np.where(np.logical_or(np.isinf(y), np.isnan(y)))[0]))
    #     pass

    print("\n\n")
