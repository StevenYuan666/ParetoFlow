import torch
from utils import set_seed

from examples.c10mop1_task import C10MOP1
from paretoflow import FlowMatching, MultipleModels, ParetoFlow, VectorFieldNet

# Set the seed
set_seed(0)

task = C10MOP1()

# If need to train the flow matching and proxies models
# Initialize the ParetoFlow sampler
pf = ParetoFlow(task=task)

# Sample the Pareto Set
res_x, res_y = pf.sample()


# If load pre-trained flow matching and proxies models
# Initialize the ParetoFlow sampler

vnet = VectorFieldNet(task.input_x.shape[1])
fm_model = FlowMatching(vnet=vnet, sigma=0.0, D=task.input_x.shape[1], T=1000)
fm_model = torch.load("saved_fm_models/c10mop1.model")

# Create the proxies model and load the saved model
proxies_model = MultipleModels(
    n_dim=task.input_x.shape[1],
    n_obj=task.input_y.shape[1],
    train_mode="Vanilla",
    hidden_size=[2048, 2048],
    save_dir="saved_proxies/",
    save_prefix="MultipleModels-Vanilla-c10mop1",
)
proxies_model.load()

pf = ParetoFlow(
    task=task,
    load_pretrained_fm=True,
    load_pretrained_proxies=True,
    fm_model=fm_model,
    proxies=proxies_model,
)

res_x, res_y = pf.sample()
gt_y = task.evaluate(res_x)
