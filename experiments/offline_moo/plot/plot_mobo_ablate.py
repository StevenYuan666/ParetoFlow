import pandas as pd

df1 = pd.read_csv("1-mobo-ablate.csv").iloc[0:, 1:]
df2 = pd.read_csv("2-mobo-ablate.csv").iloc[0:, 1:]

mean_val = (df1 + df2) / 2
mean_rank = mean_val.rank(axis=0)
print(mean_rank)

import matplotlib
import matplotlib.pyplot as plt

params = {
    "lines.linewidth": 1.5,
    "legend.fontsize": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}
matplotlib.rcParams.update(params)

import matplotlib.pyplot as plt

plt.rc("font", family="Times New Roman")

plt.figure(figsize=(8, 6))
tasks = ["50", "100", "200", "400"]
ranks = [2.25, 1.08, 3.75, 2.41]

import numpy as np

colors = [
    np.array([1.0, 0.49, 0.0]),
    np.array([1.0, 0.39, 0.1]),
    np.array([1.0, 0.29, 0.2]),
    np.array([1.0, 0.19, 0.3]),
]
x = np.arange(len(tasks))
plt.bar(x, ranks, color=colors, width=0.6)

plt.ylabel("Avg. Rank")
plt.title("Average rank over 6 tasks")

# plt.bar(x_400, ranks_400, width=width, color=np.array([0.00, 0.7, 0.15]), label='MOBO-400')
# plt.bar(x_200, ranks_200, width=width, color=np.array([0.1, 0.8, 0.35]), label='MOBO-200')
# plt.bar(x_100, ranks_100, width=width, color=np.array([0.2, 0.9, 0.55]), label='MOBO-100')
# plt.bar(x_50, ranks_50, width=width, color=np.array([0.3, 1.0, 0.75]), label='MOBO-50')
plt.xticks(x, tasks)
plt.savefig("test1.png")
plt.savefig("test-ablate.pdf")
