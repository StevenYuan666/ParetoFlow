# ParetoFlow
This repository contains the official implementation for the paper "ParetoFlow: Guided Flows in Multi-Objective Optimization". This repository is released for re-running the experiments in the paper.

Moreover, to ease any future extension of this work, we provide a pip installable package [ParetoFlow](https://pypi.org/project/paretoflow/).

## Installation

```bash
conda create -n paretoflow python=3.10
conda activate paretoflow
pip install -r requirements.txt
pip install paretoflow
```

Or Start locally:

```bash
conda create -n paretoflow python=3.10
conda activate paretoflow
pip install -r requirements.txt
git clone https://github.com/StevenYuan666/ParetoFlow.git
cd ParetoFlow
pip install -e .
```

## Usage
We accept `.py` files for input features and labels, where the continuous features has shape `(n_samples, n_dim)`, and the discrete features has shape `(n_samples, seq_len)`.
The labels are the objective values, with shape `(n_samples, n_obj)`.

When having discrete features, we need to convert the discrete features to continuous logits, as stated in the [ParetoFlow paper](https://arxiv.org/abs/2412.03718). The implementation follows the [design-bench](https://github.com/brandontrabucco/design-bench).

In our implementation, we support both z-score normalization and min-max normalization.
In our paper, we use z-score normalization for training the proxies and flow matching model. Min-max normalization is used for calculating the hypervolume, aligining with [offline-moo](https://github.com/lamda-bbo/offline-moo?tab=readme-ov-file#offline-multi-objective-optimization).

If you have your data as `x.npy` and `y.npy`, you can use the following code to train the proxies and flow matching model (continuous features for illustration, see the examples for discrete features):
```python
from paretoflow import train_proxies, train_flow_matching, FlowMatching, MultipleModels, ParetoFlowSampler, VectorFieldNet
from paretoflow import to_integers, to_logits, z_score_denormalize_x, z_score_normalize_x

# Load the data
all_x = np.load("examples/data/zdt1-x-0.npy")
all_y = np.load("examples/data/zdt1-y-0.npy")

# Normalize the data
all_x_normalized, x_mean, x_std = z_score_normalize_x(all_x)

# Train the proxies model
proxies_model = train_proxies(all_x_normalized, all_y, device="cuda")

# Train the flow matching model
fm_model = train_flow_matching(all_x_normalized, device="cuda")

# Create the sampler
sampler = ParetoFlowSampler(fm_model, proxies_model)

# Sample the data
res_x = sampler.sample(n_samples=1000)

# Denormalize the data
res_x_denormalized = z_score_denormalize_x(res_x, x_mean, x_std)

print(res_x_denormalized)
```

## Examples
```bash
python examples/train_fm.py
python examples/train_proxies.py
python examples/sampling.py
```

## Citation

If you find ParetoFlow useful in your research, please consider citing:

```bibtex
@misc{yuan2024paretoflowguidedflowsmultiobjective,
      title={ParetoFlow: Guided Flows in Multi-Objective Optimization}, 
      author={Ye Yuan and Can Chen and Christopher Pal and Xue Liu},
      year={2024},
      eprint={2412.03718},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2412.03718}, 
}
```