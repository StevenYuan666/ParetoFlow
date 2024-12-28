# ParetoFlow

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

```bash
python examples/train_fm.py
python examples/train_proxies.py
python examples/sampling.py
```

## Citation

If you find ParetoFlow useful in your research, please consider citing:

```bibtex
```
