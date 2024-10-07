import numpy as np
from utils import set_seed

from paretoflow import train_proxies

set_seed(0)

all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

saved_path, model = train_proxies(all_x, all_y, "zdt1")
