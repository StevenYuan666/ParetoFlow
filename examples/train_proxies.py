import numpy as np
from utils import set_seed

from paretoflow import train_proxies

# Set the seed
set_seed(0)

# Load the data
all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

# Train the proxies
saved_path, model = train_proxies(all_x, all_y, "zdt1")
