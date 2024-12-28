import os

import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from utils import base_path

from offline_moo.off_moo_bench.problem import get_problem


def plot_all(env_names, model_type):
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    i, j = 0, 0
    for env_name in env_names:
        if j == 5:
            j = 0
            i += 1
        ax = axs[i, j]
        results_dir = os.path.join(
            base_path,
            "results",
            env_name,
            model_type,
            "normalize-layer_and_random",
            "1",
        )
        problem = get_problem(env_name)
        nadir_point = problem.get_nadir_point()
        res_y = np.load(os.path.join(results_dir, "y_predict.npy"))
        # res_y = y_pred[NonDominatedSorting().do(y_pred)[0]]
        pop_init = np.load(os.path.join(results_dir, "pop_init.npy"))
        ax.set_title(f"{model_type} on {env_name}")
        ax.scatter(nadir_point[0], nadir_point[1], color="orange", label="nadir point")
        ax.scatter(pop_init[:, 0], pop_init[:, 1], color="pink", label="pop init")
        ax.scatter(res_y[:, 0], res_y[:, 1], color="blue", label="res pf")
        j += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f"{model_type}-n_obj=2.png")


if __name__ == "__main__":
    plot_all(
        [
            "kursawe",
            "omnitest",
            "zdt1",
            "zdt2",
            "zdt3",
            "zdt4",
            "zdt6",
            "vlmop2",
            "re21",
            "re23",
        ],
        "multi",
    )
