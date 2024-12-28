import os
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def flatten_config(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_records(exp_dir: str, table_names: list, verbose=False):
    exp_path = Path(exp_dir)
    config_files = [f.as_posix() for f in exp_path.rglob("*/.hydra/config.yaml")]

    trial_dirs = ["/".join(f.split("/")[:-2]) for f in config_files]
    trial_configs = [flatten_config(OmegaConf.load(f)) for f in config_files]
    print(f"{len(trial_dirs)} trials found")

    records = []
    for parent_dir, config in zip(trial_dirs, trial_configs):
        trial_dict = config
        for t_name in table_names:
            table_path = os.path.join(parent_dir, t_name)
            try:
                df = pd.read_csv(table_path, index_col="step")
                df.index = df.index.map(str)
                v = df.unstack().to_frame().sort_index(level=1).T
                v.columns = v.columns.map("_".join)
                trial_dict.update(v.to_dict("records")[0])
                status = "success"
            except FileNotFoundError:
                status = "not found"

            if verbose or status == "not found":
                print(f"{table_path} --> {status}")

        records.append(trial_dict)
    return records


def compound_comparison(df, df_keys, comp_values: list, comp_type):
    if isinstance(comp_type, str):
        comp_type = [comp_type] * len(comp_values)

    row_mask = np.ones(
        (
            len(
                df,
            )
        )
    ).astype(bool)
    for key, val, c_type in zip(df_keys, comp_values, comp_type):
        if c_type == "==":
            row_mask *= (df[key] == val).values
    return pd.Series(row_mask)
