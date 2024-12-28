import pickle

import numpy as np

with open("re21-none-y.pkl", "rb+") as f:
    y_none = pickle.load(f)
with open("re21-onlybest_1-y.pkl", "rb+") as f:
    y_ob = pickle.load(f)
with open("re21-none-y-srg.pkl", "rb+") as f:
    y_none_srg = pickle.load(f)
with open("re21-onlybest_1-y-srg.pkl", "rb+") as f:
    y_ob_srg = pickle.load(f)

from utils import read_raw_data

_, raw_y, _ = read_raw_data(env_name="re21", filter_type="best", return_rank=False)

y_max = np.max(raw_y, axis=0)
y_min = np.min(raw_y, axis=0)


def normalize_y(y):
    return (y - y_min) / (y_max - y_min)


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

colormap = cm.get_cmap(name="YlGnBu")
n_gen = 30
c = [colormap(i) for i in np.linspace(0, 1, n_gen)]
myCycler = cycler(color=c)
plt.gca().set_prop_cycle(myCycler)

figs, axes = plt.subplots(2, 2, figsize=(13, 13))
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
for i in range(n_gen):
    alpha = i / n_gen  # Adjust alpha for gradient effect
    y_n = y_none[i][:, :]
    y_o = y_ob[i][:, :]
    y_n_s = y_none_srg[i]
    y_o_s = y_ob_srg[i]
    y_n = normalize_y(y_n)
    y_o = normalize_y(y_o)
    y_n_s = normalize_y(y_n_s)
    y_o_s = normalize_y(y_o_s)
    ax3.scatter(y_n[:, :, 0], y_n[:, :, 1], color=colormap(alpha), alpha=1.0)
    ax4.scatter(y_o[:, :, 0], y_o[:, :, 1], color=colormap(alpha), alpha=1.0)
    ax1.scatter(y_n_s[:, :, 0], y_n_s[:, :, 1], color=colormap(alpha), alpha=1.0)
    ax2.scatter(y_o_s[:, :, 0], y_o_s[:, :, 1], color=colormap(alpha), alpha=1.0)

ax1.set_xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
ax1.set_ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})
ax2.set_xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
ax2.set_ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})
ax3.set_xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
ax3.set_ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})
ax4.set_xlabel(r"$f_1$", fontdict={"family": "Times New Roman"})
ax4.set_ylabel(r"$f_2$", fontdict={"family": "Times New Roman"})
ax1.set_title("Multi-Head w/o data pruning", fontdict={"family": "Times New Roman"})
ax2.set_title("Multi-Head w/ data pruning", fontdict={"family": "Times New Roman"})

plt.savefig("test.png")
plt.savefig("Multi-Head-DataPruning-RE21.pdf")
assert 0
