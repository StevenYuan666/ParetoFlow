import os

import numpy as np

envs = [
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz5",
    "dtlz6",
    "dtlz7",
    "zdt1",
    "zdt2",
    "zdt3",
    "omnitest",
    "vlmop2",
    "vlmop3",
    "mo_nas",
    "mo_swimmer_v2",
    "mo_hopper_v2",
    "mo_tsp",
    "mo_cvrp",
    "mo_kp",
    "molecule",
    "rfp",
    "regex",
    "re21",
    "re23",
    "re34",
    "re37",
]

nadir_points = {
    "molecule": np.array([-0.0, -0.0]),
    "regex": np.array([0.64954899, 0.7886475, 0.73789501]),
    "rfp": np.array([4.0, 4.0]),
}

env2name = {
    "zdt1": "ZDT1",
    "zdt2": "ZDT2",
    "zdt3": "ZDT3",
    "omnitest": "OmniTest",
    "vlmop2": "VLMOP2",
    "mo_hopper_v2": "MO-Hopper-V2",
    "mo_swimmer_v2": "MO-Swimmer-V2",
    "mo_cvrp": "MO-CVRP",
    "mo_tsp": "MO-TSP",
    "mo_kp": "MO-KP",
    "molecule": "Molecule",
    "rfp": "RFP",
    "re21": "RE21",
    "re23": "RE23",
    "dtlz1": "DTLZ1",
    "dtlz2": "DTLZ2",
    "dtlz3": "DTLZ3",
    "dtlz5": "DTLZ5",
    "dtlz6": "DTLZ6",
    "dtlz7": "DTLZ7",
    "vlmop3": "VLMOP3",
    "mo_nas": "MO-NAS",
    "regex": "Regex",
    "re34": "RE34",
    "re37": "RE37",
}

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, cycler

params = {
    "lines.linewidth": 1.5,
    "legend.fontsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 25,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
}
matplotlib.rcParams.update(params)

plt.rc("font", family="Times New Roman")


def plot_y(y, ax, pareto_front=None, nadir_point=None, d_best=None):
    n_obj = len(y[0])
    if n_obj == 2:
        if pareto_front is not None:
            ax.scatter(pareto_front[:, 0], pareto_front[:, 1], color="red")
        if nadir_point is not None:
            ax.scatter(nadir_point[0], nadir_point[1], color="green")
        if d_best is not None:
            ax.scatter(d_best[:, 0], d_best[:, 1], color="blue")
        plt.scatter(y[:, 0], y[:, 1], color="red")
    elif n_obj == 3:
        # ax = fig.add_subplot(111, projection='3d')

        if pareto_front is not None:
            ax.scatter(
                pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2], color="red"
            )
        if nadir_point is not None:
            ax.scatter(nadir_point[0], nadir_point[1], nadir_point[2], color="green")
        if d_best is not None:
            ax.scatter(d_best[:, 0], d_best[:, 1], d_best[:, 2], color="blue")
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], color="red")

    else:
        raise NotImplementedError


fig = plt.figure(figsize=(25, 25))


def normalize_y(y, y_max, y_min):
    return (y - y_min) / (y_max - y_min)


for i in range(5):
    for j in range(5):
        # ax = axes[i, j]
        env = envs[i * 5 + j]
        results_dir = os.path.join(
            base_path,
            env,
            "multi_head",
            "final-normalize-best-onlybest_1-grad_norm",
            "1",
        )
        assert os.path.exists(results_dir)

        try:
            nadir_point = nadir_points[env]
        except:
            from offline_moo.off_moo_bench.problem import get_problem

            problem = get_problem(env)
            nadir_point = problem.get_nadir_point()

        if len(nadir_point) == 2:
            ax = fig.add_subplot(5, 5, i * 5 + j + 1)
        elif len(nadir_point) == 3:
            ax = fig.add_subplot(5, 5, i * 5 + j + 1, projection="3d")
        else:
            raise NotImplementedError

        from utils import read_raw_data

        _, y_raw, _ = read_raw_data(env_name=env, filter_type="best", return_rank=False)
        y_max = np.max(y_raw, axis=0)
        y_min = np.min(y_raw, axis=0)

        pop_init = np.load(os.path.join(results_dir, "pop_init.npy"))
        y_pred = np.load(os.path.join(results_dir, "y_predict.npy"))

        pop_init = normalize_y(pop_init, y_max, y_min)
        y_pred = normalize_y(y_pred, y_max, y_min)
        nadir_point = normalize_y(nadir_point, y_max, y_min)

        nadir_point = (
            nadir_point * 1.1
            if env not in ["mo_hopper_v2", "mo_swimmer_v2"]
            else nadir_point * 2
        )

        plot_y(y_pred, ax, d_best=pop_init, nadir_point=nadir_point)

        # print(pop_init)
        # print(y_pred)
        # print(nadir_point)
        ax.set_title(f"{env2name[env]}")
        if len(nadir_point) == 2:
            ax.set_xlabel("$f_1$")
            ax.set_ylabel("$f_2$")

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("pf.png")
plt.savefig("pf-multi-head-grad-norm.pdf")
