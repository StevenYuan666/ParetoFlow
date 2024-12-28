import os

import numpy as np
from utils import read_data, read_raw_data

base_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "generate_instance"
)
envs = ["regex", "rfp"]
# envs = ['zdt1', 'zdt2', 'zdt3', 'dtlz1', 'dtlz2', 'dtlz3' ,'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
env2y = {
    "mo_kp": os.path.join(base_path, "y_all_kp.npy"),
    "mo_tsp": os.path.join(base_path, "y_all_tsp.npy"),
    "mo_cvrp": os.path.join(base_path, "y_all_cvrp.npy"),
}
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
i, j = 0, 0
for i in range(1):
    for j in range(2):
        env = envs[i * 2 + j]
        print(env)
        ax = axs[i * 2 + j]

        _, raw_y, _ = read_raw_data(
            env, filter_type="best", return_x=False, return_rank=False
        )
        _, y, _ = read_data(env, filter_type="best", return_x=False, return_rank=False)
        print(y.shape)

        # y_all = np.load(env2y[env])
        # pf = y_all[-256:, :]

        if env == "mo_tsp":
            # from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
            # fronts = NonDominatedSorting().do(raw_y, return_rank=True, n_stop_if_ranked=256)[0]
            # indices_cnt = 0
            # indices_select = []
            # for front in fronts:
            #     if indices_cnt + len(front) < 256:
            #         indices_cnt += len(front)
            #         indices_select += [int(i) for i in front]
            #     else:
            #         idx = np.random.randint(len(front), size=(256-indices_cnt, ))
            #         indices_select += [int(i) for i in front[idx]]
            #         break
            from utils import get_N_nondominated_index

            indices_select = get_N_nondominated_index(raw_y, 256)
            pf = raw_y[indices_select]

        # try:
        #     from m2bo_bench.problem import get_problem
        #     pf = get_problem(env).get_pareto_front()

        # ax.scatter(pf[:, 0], pf[:, 1], color='red', label='pareto_front')
        # except:
        #     pass
        ax.scatter(raw_y[:, 0], raw_y[:, 1], color="pink", label="full data")
        ax.scatter(y[:, 0], y[:, 1], color="blue", label="y")
        # ax.scatter(pf[:, 0], pf[:, 1], color='red', label='pareto_front')

        ax.set_title(f"{env} dataset")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        # ax.legend()
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("dataset_visualization_molecule.png")
