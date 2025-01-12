# ParetoFlow
ParetoFlow is a Python package for offline multi-objective optimization using Generative Flow Models with Multi Predictors Guidance to approximate the Pareto front.

## Installation

```bash
conda create -n paretoflow python=3.10
conda activate paretoflow
pip install paretoflow
```

Or Start locally:

```bash
conda create -n paretoflow python=3.10
conda activate paretoflow
git clone https://github.com/StevenYuan666/ParetoFlow.git
cd ParetoFlow
pip install -e .
```

## Usage
We accept `.npy` files for input features and labels, where the continuous features has shape `(n_samples, n_dim)`, and the discrete features has shape `(n_samples, seq_len)`.
The labels are the objective values, with shape `(n_samples, n_obj)`.

When having discrete features, we need to convert the discrete features to continuous logits, as stated in the [ParetoFlow paper](https://arxiv.org/abs/2412.03718). The implementation follows the [design-bench](https://github.com/brandontrabucco/design-bench).

In our implementation, we support both z-score normalization and min-max normalization.
In our paper, we use z-score normalization for training the proxies and flow matching model. Min-max normalization is used for calculating the hypervolume, aligining with [offline-moo](https://github.com/lamda-bbo/offline-moo?tab=readme-ov-file#offline-multi-objective-optimization).

If you have your data as `x.npy` and `y.npy`, you can use the following code to define a new task (a new optimization problem you want to solve), we use continuous features for illustration, see the `examples/c10mop1_task.py` for discrete features example:
```python
import numpy as np
from paretoflow import Task

class ZDT2(Task):
    def __init__(self):
        # Load the data
        all_x = np.load("examples/data/zdt2-x-0.npy")
        all_y = np.load("examples/data/zdt2-y-0.npy")
        super().__init__(
            task_name="ZDT2",
            input_x=all_x,
            input_y=all_y,
            x_lower_bound=np.array([0.0] * all_x.shape[1]),
            x_upper_bound=np.array([1.0] * all_x.shape[1]),
            nadir_point=np.array([0.99999706, 9.74316166]),
        )

    def evaluate(self, x):
        """
        This is only for illustrataion purpose, we omit the evaluation function in this example. 
        See offline-moo benchmark for more details about the evaluation function for ZDT2. 
        Or one can use the `get_problem` function in the `pymoo` package to evaluate the ZDT2 problem.
        """
        pass

```

Once you have defined the task, you can use the following code to train the flow matching and proxies models:
```python
import torch
from utils import set_seed
from paretoflow import FlowMatching, MultipleModels, ParetoFlow, VectorFieldNet
from examples.zdt2_task import ZDT2

# Set the seed
set_seed(0)
# Instantiate the task
task = ZDT2()
# Initialize the ParetoFlow sampler
pf = ParetoFlow(task=task) # This will automatically train the flow matching and proxies
# Sample the Pareto Set
res_x, res_y = pf.sample()
# Evaluate the Pareto Set
gt_y = task.evaluate(res_x)
```

Or you can load the pre-trained flow matching and proxies models:
```python
import numpy as np
import torch
from paretoflow import ParetoFlow, VectorFieldNet, FlowMatching, MultipleModels
from examples.zdt2_task import ZDT2

# Set the seed
set_seed(0)
# Instantiate the task
task = ZDT2()

# If load pre-trained flow matching and proxies models
# Initialize the ParetoFlow sampler
vnet = VectorFieldNet(task.input_x.shape[1])
fm_model = FlowMatching(vnet=vnet, sigma=0.0, D=task.input_x.shape[1], T=1000)
fm_model = torch.load("saved_fm_models/ZDT2.model")

# Create the proxies model and load the saved model
proxies_model = MultipleModels(
    n_dim=task.input_x.shape[1],
    n_obj=task.input_y.shape[1],
    train_mode="Vanilla",
    hidden_size=[2048, 2048],
    save_dir="saved_proxies/",
    save_prefix="MultipleModels-Vanilla-ZDT2",
)
proxies_model.load()

pf = ParetoFlow(
    task=task,
    load_pretrained_fm=True,
    load_pretrained_proxies=True,
    fm_model=fm_model,
    proxies=proxies_model,
)

res_x, predicted_res_y = pf.sample()
gt_y = task.evaluate(res_x)
```

**More Importantly**, we also allow users to pass in their own pretrained flow matching and proxies models. We require the flow matching model to be a `nn.Module` object and also pass in two key arguments `vnet` and `time_embedding`, which are both `nn.Module` objects. The `vnet` is the network approximation for the vector field in the flow matching model, and the `time_embedding` is a mapping from continuous time between [0, 1] to the embedding space. See more details in the docstrings of the `ParetoFlow` class.


## Future Works
- [ ] Refactor ParetoFlow as an optimization algorithm in the `pymoo` package.
- [ ] Support using ParetoFlow on `problems` in the `pymoo` package.
- [ ] Merge ParetoFlow with the `pymoo` package.

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
