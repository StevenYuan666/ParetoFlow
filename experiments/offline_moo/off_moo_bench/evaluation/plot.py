import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_y(y, save_dir, config, pareto_front=None, nadir_point=None, d_best=None):
    params = {
        "lines.linewidth": 1.5,
        "legend.fontsize": 18,
        "axes.labelsize": 20,
        "axes.titlesize": 25,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
    matplotlib.rcParams.update(params)

    plt.rc("font", family="Times New Roman")

    n_obj = len(y[0])

    if n_obj == 2:
        plt.figure(figsize=(10, 8))
        if pareto_front is not None:
            plt.scatter(
                pareto_front[:, 0],
                pareto_front[:, 1],
                color="red",
                label="Pareto Front",
            )
        if nadir_point is not None:
            plt.scatter(
                nadir_point[0], nadir_point[1], color="green", label="Nadir Point"
            )
        if d_best is not None:
            plt.scatter(
                d_best[:, 0], d_best[:, 1], color="pink", label="$\\mathcal{D}$(best)"
            )
        plt.scatter(y[:, 0], y[:, 1], color="blue", label="Solu. Performance")
        plt.xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
        plt.ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})

    elif n_obj == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if pareto_front is not None:
            ax.scatter(
                pareto_front[:, 0],
                pareto_front[:, 1],
                pareto_front[:, 2],
                color="red",
                label="Pareto Front",
            )
        if nadir_point is not None:
            ax.scatter(
                nadir_point[0],
                nadir_point[1],
                nadir_point[2],
                color="green",
                label="Nadir Point",
            )
        if d_best is not None:
            ax.scatter(
                d_best[:, 0],
                d_best[:, 1],
                d_best[:, 2],
                color="pink",
                label="$\\mathcal{D}$(best)",
            )
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], color="blue", label="Solu. Performance")

        ax.set_ylim([np.max(y[:, 1]), np.min(y[:1])])

        ax.set_xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
        ax.set_ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})
        ax.set_zlabel(r"$f_2$", fontdict={"family": "Times New Roman"})

    else:
        fig, axs = plt.subplots(n_obj, n_obj, figsize=(20, 20))
        for i in range(n_obj):
            for j in range(n_obj):
                if i == j:
                    continue
                ax = axs[i, j]
                ax.set_title(
                    f"obj.{i + 1} and obj.{j + 1}",
                    fontdict={"family": "Times New Roman"},
                )
                ax.set_xlabel(f"f{i}", fontdict={"family": "Times New Roman"})
                ax.set_ylabel(f"f{j}", fontdict={"family": "Times New Roman"})
                if pareto_front is not None:
                    ax.scatter(
                        pareto_front[:, i],
                        pareto_front[:, j],
                        color="red",
                        label="Pareto Front",
                    )
                if nadir_point is not None:
                    ax.scatter(
                        nadir_point[i],
                        nadir_point[j],
                        color="green",
                        label="Nadir Point",
                    )
                if d_best is not None:
                    ax.scatter(
                        d_best[:, 0],
                        d_best[:, 1],
                        color="pink",
                        label="$\\mathcal{D}$(best)",
                    )
                ax.scatter(y[:, i], y[:, j], color="blue", label="Solu. Performance")

        plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.title(f"Results of {config['task']}", fontdict={"family": "Times New Roman"})
    plt.legend()

    plt.savefig(os.path.join(save_dir, "pareto_front.png"))
    plt.savefig(os.path.join(save_dir, "pareto_front.pdf"))
