from pathlib import Path

import numpy as np
import pandas as pd


def get_gaussian_region(mean, variance):
    lb = mean - 2 * np.sqrt(variance)
    ub = mean + 2 * np.sqrt(variance)
    return mean, lb, ub


def combine_trials(df_list, x_col):
    df_list = sorted(df_list, key=lambda x: len(x), reverse=True)
    merged_df = df_list[0]
    if not all([len(df) == len(merged_df) for df in df_list]):
        print("trials are not all the same length!")
    for df in df_list[1:]:
        merged_df = pd.merge_asof(merged_df, df, on=x_col, direction="nearest")
    merged_df.fillna(method="ffill")
    return merged_df


def get_arm(
    exp_dir, arm_name, table_name, x_col, y_col, window=1, transform=None, verbose=False
):
    arm_path = Path(exp_dir) / arm_name
    arm_dfs = [pd.read_csv(f) for f in arm_path.rglob(f"*{table_name}*")]
    if verbose:
        print(f"{len(arm_dfs)} tables found in {arm_path.as_posix()}")

    merged_df = combine_trials(arm_dfs, x_col)
    yval_df = merged_df.filter(regex=f"^{y_col}", axis=1)

    x_range = merged_df[x_col].values
    arm_data = yval_df.rolling(window, min_periods=1).mean().values
    arm_data = arm_data if transform is None else transform(arm_data)

    mean, lb, ub = get_gaussian_region(arm_data.mean(-1), arm_data.var(-1))
    return x_range, mean, lb, ub


def draw_arm_comparison(
    ax,
    root_dir,
    arms,
    table_name,
    x_key,
    y_key,
    transform=None,
    window=1,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    linewidth=2,
    alpha=0.3,
):
    for i, (arm_key, arm_path) in enumerate(arms.items()):
        x_range, mean, lb, ub = get_arm(
            root_dir, arm_path, table_name, x_key, y_key, window, transform=transform
        )
        ax.plot(x_range, mean, linewidth=linewidth)
        ax.fill_between(x_range, lb, ub, alpha=alpha, label=arm_key)

    xlabel = x_key if xlabel is None else xlabel
    ylabel = y_key if ylabel is None else ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)

    return ax
