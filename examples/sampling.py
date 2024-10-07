import numpy as np
import torch
from utils import set_seed

from paretoflow import FlowMatching, MultipleModels, ParetoFlowSampler, VectorFieldNet

set_seed(0)

all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vnet = VectorFieldNet(all_x.shape[1])
fm_model = FlowMatching(vnet=vnet, sigma=0.0, D=all_x.shape[1], T=1000)
fm_model = torch.load("saved_fm_models/zdt1.model")

proxies_model = MultipleModels(
    n_dim=all_x.shape[1],
    n_obj=all_y.shape[1],
    train_mode="Vanilla",
    hidden_size=[2048, 2048],
    save_dir="saved_proxies/",
    save_prefix="MultipleModels-Vanilla-zdt1",
)
proxies_model.load()

sampler = ParetoFlowSampler(
    fm_model=fm_model,
    proxies=proxies_model,
)

res_x, res_y = sampler.sample(
    all_x=all_x,
    all_y=all_y,
    xl=np.array([0.0] * all_x.shape[1]),
    xu=np.array([1.0] * all_x.shape[1]),
)

print(len(res_x))
