import numpy as np
import torch
from utils import set_seed

from paretoflow import train_flow_matching

# Set the seed
set_seed(0)

# Load the data
all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the generative flow matching model
val_loss, saved_path, model = train_flow_matching(
    all_x, device, "zdt1", validation_size=1000
)