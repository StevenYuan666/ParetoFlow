import numpy as np
from utils import set_seed

from paretoflow import (
    to_integers,
    to_logits,
    train_proxies,
    z_score_denormalize_x,
    z_score_normalize_x,
)

# Set the seed
set_seed(0)

# Continuous features
# Load the data
all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

# Normalize the data
all_x_normalized, x_mean, x_std = z_score_normalize_x(all_x)
all_x_denormalized = z_score_denormalize_x(all_x_normalized, x_mean, x_std)
assert np.allclose(all_x_denormalized, all_x, atol=1e-6)

# Train the proxies
saved_path, model = train_proxies(all_x_normalized, all_y, "zdt1")

# Discrete features
# Load the data
all_x = np.load("examples/data/c10mop1-x-0.npy")
all_y = np.load("examples/data/c10mop1-y-0.npy")

# Convert the discrete features to logits
all_x_logits = to_logits(
    all_x,
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
)
all_x_integers = to_integers(
    all_x_logits,
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
)
assert np.all(all_x_integers == all_x)

# Normalize the data
all_x_normalized, x_mean, x_std = z_score_normalize_x(all_x_logits)
all_x_denormalized = z_score_denormalize_x(all_x_normalized, x_mean, x_std)
assert np.allclose(all_x_denormalized, all_x_logits, atol=1e-6)

# Train the proxies
saved_path, model = train_proxies(all_x_normalized, all_y, "c10mop1")
