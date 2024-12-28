import random

import hydra
import randomname
import torch
from omegaconf import DictConfig, OmegaConf
from upcycle.random.seed import set_all_seeds


def startup(hydra_cfg):
    trial_id = hydra_cfg.trial_id
    if hydra_cfg.job_name is None:
        hydra_cfg.job_name = "_".join(
            randomname.get_name().lower().split("-") + [str(trial_id)]
        )
    hydra_cfg.seed = (
        random.randint(0, 100000) if hydra_cfg.seed is None else hydra_cfg.seed
    )
    set_all_seeds(hydra_cfg.seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(
        hydra_cfg, resolve=True
    )  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    print(OmegaConf.to_yaml(hydra_cfg))
    with open("hydra_config.txt", "w") as f:
        f.write(OmegaConf.to_yaml(hydra_cfg))
    print(f"GPU available: {torch.cuda.is_available()}")
    return hydra_cfg, logger
