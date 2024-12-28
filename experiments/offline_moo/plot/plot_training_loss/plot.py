import matplotlib
import matplotlib.pyplot as plt
import numpy as np

params = {
    "lines.linewidth": 1.5,
    "legend.fontsize": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}
matplotlib.rcParams.update(params)

plt.rc("font", family="Times New Roman")


# fig = plt.figure(figsize=(13, 13))
nas_wo = np.load("mo_nas_wo_loss.npy")
nas_gn = np.load("mo_nas_gn_loss.npy")
# ax = fig.add_subplot(2, 2, 1)
plt.figure(figsize=(8, 6))
plt.plot(
    range(len(nas_wo)),
    nas_wo,
    color=np.array([1.0, 0.25, 0.33]),
    label="Vallina Multi-Head",
)
plt.plot(
    range(len(nas_gn)),
    nas_gn,
    color=np.array([0.19, 0.55, 0.91]),
    label="GradNorm Multi-Head",
)
plt.title("Elites Loss on MO-NAS")
plt.xlabel("# epochs")
plt.ylabel("Elite loss")
plt.legend(loc="upper right")
plt.savefig("2.png")
plt.savefig("2.pdf")

dtlz1_wo = np.load("dtlz1_wo_loss.npy")
dtlz1_gn = np.load("dtlz1_gn_loss.npy")
# ax = fig.add_subplot(2, 2, 2)
plt.figure(figsize=(8, 6))
plt.plot(
    range(len(dtlz1_wo)),
    dtlz1_wo,
    color=np.array([1.0, 0.25, 0.33]),
    label="Vallina Multi-Head",
)
plt.plot(
    range(len(dtlz1_gn)),
    dtlz1_gn,
    color=np.array([0.19, 0.55, 0.91]),
    label="GradNorm Multi-Head",
)
plt.title("Elites Loss on DTLZ1")
plt.xlabel("# epochs")
plt.ylabel("Elite loss")
plt.legend(loc="upper right")
plt.savefig("1.png")
plt.savefig("1.pdf")


import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import read_raw_data


def normalize_y(y, env_name):
    _, y_raw, _ = read_raw_data(
        env_name=env_name, filter_type="best", return_rank=False
    )
    y_max = np.max(y_raw, axis=0)
    y_min = np.min(y_raw, axis=0)
    return (y - y_min) / (y_max - y_min)


nas_wo_y = np.load("mo_nas_wo_y.npy")
nas_gn_y = np.load("mo_nas_gn_y.npy")
nas_wo_y = normalize_y(nas_wo_y, "mo_nas")
nas_gn_y = normalize_y(nas_gn_y, "mo_nas")
# ax = fig.add_subplot(2, 2, 3, projection='3d')
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    nas_wo_y[:, 0],
    nas_wo_y[:, 1],
    nas_wo_y[:, 2],
    color=np.array([1.0, 0.25, 0.33]),
    label="Vallina",
)
ax.scatter(
    nas_gn_y[:, 0],
    nas_gn_y[:, 1],
    nas_gn_y[:, 2],
    color=np.array([0.19, 0.55, 0.91]),
    label="GradNorm",
)
plt.title("Pareto Fronts on MO-NAS")
ax.legend(loc="upper right")
plt.savefig("4.png")
plt.savefig("4.pdf")

dtlz1_wo_y = np.load("dtlz1_wo_y.npy")
dtlz1_gn_y = np.load("dtlz1_gn_y.npy")
dtlz1_wo_y = normalize_y(dtlz1_wo_y, "dtlz1")
dtlz1_gn_y = normalize_y(dtlz1_gn_y, "dtlz1")
# ax = fig.add_subplot(2, 2, 4, projection='3d')
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    dtlz1_wo_y[:, 0],
    dtlz1_wo_y[:, 1],
    dtlz1_wo_y[:, 2],
    color=np.array([1.0, 0.25, 0.33]),
    label="Vallina",
)
ax.scatter(
    dtlz1_gn_y[:, 0],
    dtlz1_gn_y[:, 1],
    dtlz1_gn_y[:, 2],
    color=np.array([0.19, 0.55, 0.91]),
    label="GradNorm",
)
plt.title("Pareto Fronts on DTLZ1")
ax.legend(loc="upper right")
plt.savefig("3.png")
plt.savefig("3.pdf")

plt.subplots_adjust(wspace=0.3, hspace=0.35)

plt.savefig("test.png")
plt.savefig("test-loss-vs-pf.pdf")
