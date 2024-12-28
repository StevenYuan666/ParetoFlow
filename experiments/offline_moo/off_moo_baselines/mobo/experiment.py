import datetime
import json
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import wandb

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(BASE_PATH)

from pymoo.algorithms.moo.nsga2 import NSGA2
from utils import get_quantile_solutions, set_seed

import offline_moo.off_moo_bench as ob
from offline_moo.off_moo_baselines.data import get_dataloader
from offline_moo.off_moo_baselines.mobo import get_mobo_solver
from offline_moo.off_moo_baselines.mobo.mobo_utils import tkwargs
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.off_moo_bench.evaluation.plot import plot_y
from offline_moo.off_moo_bench.task_set import *


def run(config: dict):
    if config["task"] in ALLTASKSDICT.keys():
        config["task"] = ALLTASKSDICT[config["task"]]

    results_dir = os.path.join(
        config["results_dir"],
        f"{config['model']}-{config['train_mode']}-{config['task']}",
    )
    config["results_dir"] = results_dir

    ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
    ts_name = f"-ts-{ts.year}-{ts.month}-{ts.day}_{ts.hour}-{ts.minute}-{ts.second}"
    run_name = f"{config['model']}-{config['train_mode']}-seed{config['seed']}-{config['task']}"

    logging_dir = os.path.join(config["results_dir"], run_name + ts_name)
    os.makedirs(logging_dir, exist_ok=True)

    if config["use_wandb"]:
        if "wandb_api" in config.keys():
            wandb.login(key=config["wandb_api"])

        wandb.init(
            project="Offline-MOO",
            name=run_name + ts_name,
            config=config,
            group=f"{config['model']}-{config['train_mode']}",
            job_type=config["run_type"],
            mode="online",
            dir=os.path.join(config["results_dir"], ".."),
        )

    with open(os.path.join(logging_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)

    set_seed(config["seed"])

    task = ob.make(config["task"])

    X = task.x.copy()
    y = task.y.copy()

    if config["data_pruning"]:
        X, y = task.get_N_non_dominated_solutions(
            N=int(X.shape[0] * config["data_preserved_ratio"]),
            return_x=True,
            return_y=True,
        )

    X_test = task.x_test.copy()
    y_test = task.y_test.copy()

    if config["to_logits"]:
        assert task.is_discrete
        task.map_to_logits()
        X = task.to_logits(X)
        X_test = task.to_logits(X_test)
    if config["normalize_xs"]:
        task.map_normalize_x()
        X = task.normalize_x(X)
        X_test = task.normalize_x(X_test)
    if config["normalize_ys"]:
        task.map_normalize_y()
        y = task.normalize_y(y)
        y_test = task.normalize_y(y_test)

    if config["to_logits"]:
        data_size, n_dim, n_classes = tuple(X.shape)
        X = X.reshape(-1, n_dim * n_classes)
        X_test = X_test.reshape(-1, n_dim * n_classes)
    else:
        data_size, n_dim = tuple(X.shape)
    n_obj = y.shape[1]

    model_save_dir = config["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)

    MOBO_Solver = get_mobo_solver(config["train_mode"])(
        config=config,
        X_init=torch.Tensor(X).to(**tkwargs),
        Y_init=torch.Tensor(y).to(**tkwargs),
        solver_kwargs={
            "ref_point": torch.Tensor(1.1 * task.nadir_point).to(**tkwargs),
            "xl": task.xl,
            "xu": task.xu,
        },
        train_gp_data_size=config["train_gp_data_size"],
        output_size=config["num_solutions"],
        negate=True,
    )

    res_x = MOBO_Solver.run()
    if config["to_logits"]:
        res_x = res_x.reshape(-1, n_dim, n_classes)
    if config["normalize_xs"]:
        task.map_denormalize_x()
        res_x = task.denormalize_x(res_x)
    if config["to_logits"]:
        task.map_to_integers()
        res_x = task.to_integers(res_x)

    res_y = task.predict(res_x)
    visible_masks = np.ones(len(res_y))
    visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
    visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
    res_x = res_x[np.where(visible_masks == 1)[0]]
    res_y = res_y[np.where(visible_masks == 1)[0]]

    res_y_75_percent = get_quantile_solutions(res_y, 0.75)
    res_y_50_percent = get_quantile_solutions(res_y, 0.50)

    nadir_point = task.nadir_point
    if config["normalize_ys"]:
        res_y = task.normalize_y(res_y)
        nadir_point = task.normalize_y(nadir_point)
        res_y_50_percent = task.normalize_y(res_y_50_percent)
        res_y_75_percent = task.normalize_y(res_y_75_percent)

    _, d_best = task.get_N_non_dominated_solutions(
        N=config["num_solutions"], return_x=False, return_y=True
    )

    np.save(file=os.path.join(logging_dir, "res_x.npy"), arr=res_x)
    np.save(file=os.path.join(logging_dir, "res_y.npy"), arr=res_y)
    plot_y(
        res_y,
        save_dir=logging_dir,
        config=config,
        nadir_point=nadir_point,
        d_best=d_best,
    )

    d_best_hv = hv(nadir_point, d_best, config["task"])
    hv_value = hv(nadir_point, res_y, config["task"])
    hv_value_50_percentile = hv(nadir_point, res_y_50_percent, config["task"])
    hv_value_75_percentile = hv(nadir_point, res_y_75_percent, config["task"])

    print(f"Hypervolume (100th): {hv_value:4f}")
    print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
    print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
    print(f"Hypervolume (D(best)): {d_best_hv:4f}")

    hv_results = {
        "hypervolume/D(best)": d_best_hv,
        "hypervolume/100th": hv_value,
        "hypervolume/75th": hv_value_75_percentile,
        "hypervolume/50th": hv_value_50_percentile,
        "evaluation_step": 1,
    }

    df = pd.DataFrame([hv_results])
    filename = os.path.join(logging_dir, "hv_results.csv")
    df.to_csv(filename, index=False)

    if config["use_wandb"]:
        wandb.log(hv_results)


if __name__ == "__main__":
    from utils import process_args

    config = process_args(return_dict=True)

    results_dir = os.path.join(BASE_PATH, "results")
    model_save_dir = os.path.join(BASE_PATH, "model")

    config["results_dir"] = results_dir
    config["model_save_dir"] = model_save_dir

    run(config)
