import numpy as np
import torch
from utils import set_seed

from paretoflow import FlowMatching, MultipleModels, ParetoFlow, VectorFieldNet

# Set the seed
set_seed(0)

# Load the data
all_x = np.load("examples/data/c10mop1-x-0.npy")
all_y = np.load("examples/data/c10mop1-y-0.npy")

# If need to train the flow matching and proxies models
# Initialize the ParetoFlow sampler
pf = ParetoFlow(
    task_name="c10mop1",
    input_x=all_x,
    input_y=all_y,
    x_lower_bound=np.array([0.0] * all_x.shape[1]),
    # This is the number of discrete values for each dimension
    x_upper_bound=np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    ),
    is_discrete=True,
)

# Sample the Pareto Set
res_x, res_y = pf.sample()

print(len(res_x))


# If load pre-trained flow matching and proxies models
# Initialize the ParetoFlow sampler

vnet = VectorFieldNet(all_x.shape[1])
fm_model = FlowMatching(vnet=vnet, sigma=0.0, D=all_x.shape[1], T=1000)
fm_model = torch.load("saved_fm_models/zdt1.model")

# Create the proxies model and load the saved model
proxies_model = MultipleModels(
    n_dim=all_x.shape[1],
    n_obj=all_y.shape[1],
    train_mode="Vanilla",
    hidden_size=[2048, 2048],
    save_dir="saved_proxies/",
    save_prefix="MultipleModels-Vanilla-zdt1",
)
proxies_model.load()

pf = ParetoFlow(
    task_name="zdt1",
    input_x=all_x,
    input_y=all_y,
    x_lower_bound=np.array([0.0] * all_x.shape[1]),
    x_upper_bound=np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    ),
    is_discrete=True,
    load_pretrained_fm=True,
    load_pretrained_proxies=True,
    fm_model=fm_model,
    proxies=proxies_model,
)

res_x, res_y = pf.sample()

print(len(res_x))
