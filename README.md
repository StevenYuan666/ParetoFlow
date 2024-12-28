# ParetoFlow
This repository contains the official implementation for the paper **"ParetoFlow: Guided Flows in Multi-Objective Optimization"**. This repository is released for re-running the experiments in the paper. See **Reproduce the experiments in the paper** for more details.

Moreover, to ease any future extension of this work, we provide a pip installable package [ParetoFlow](https://pypi.org/project/paretoflow/). See **Use ParetoFlow as a pip package** for more details.

**Folder Structure**
```bash
|-- examples/ # Examples for ParetoFlow
|-- experiments/ # Reproduce the experiments in the paper
    |-- offline_moo/ # benchmark for offline MOO
    |-- scripts/ # scripts
    |-- gfmo_args.py # arguments
    |-- gfmo_experiments.py # experiments
    |-- gfmo_net.py # networks
    |-- gfmo_reference_directions.py # reference directions
    |-- gfmo_utils.py # utils
    |-- gfmo.py # main\
    |-- requirements.txt # Dependencies for reproducing the experiments
|-- paretoflow/ # ParetoFlow as a pip package
|-- saved_fm_models/ # Saved flow matching models
|-- saved_proxies/ # Saved proxies models
|-- requirements.txt # Dependencies for ParetoFlow package
|-- README.md
|-- LICENSE
|-- .gitignore
|-- setup.py # Setup for ParetoFlow package
```

## Reproduce the experiments in the paper
### Install Dependencies

We refer most of following steps to the README of the [offline MOO benchmark](https://github.com/lamda-bbo/offline-moo?tab=readme-ov-file#evoxbench).

#### Create a Virtual Environment
```bash
conda create -n gfmo python=3.8
conda activate gfmo
conda install gxx_linux-64 gcc_linux-64
conda install --channel=conda-forge libxcrypt
pip install -r experiments/requirements.txt
bash experiments/offline_moo/fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/gfmo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"
```

#### Data Download
Download the offline dataset provided by the offline MOO benchmark [here](https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT).

Put all downloaded datasets into `experiments/offline_moo/data/`

#### Download and Install FoldX
1. Download FoldX Emulator [here](https://foldxsuite.crg.eu/academic-license-info).
2. Copy the contents of the downloaded archive to `~/foldx`. 
3. `cd experiments/offline_moo/off_moo_bench/problem/lambo/` Go to the directory of lambo.
4. Run `pip install -e .`.
5. Run `python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=proxy_rfp tokenizer=protein` to generate an instance `proxy_rfp_problem.pkl` of RFP task.

#### Download and Install EvoXBench
1. Download the database [here](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view) and the datasaet [here](https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view).
2. Save them and move the unziped files to `experiments/offline_moo/off_moo_bench/problem/mo_nas/database` and `experiments/offline_moo/off_moo_bench/problem/mo_nas/data`.
3. Run `pip install evoxbench`.
4. Run `python experiments/offline_moo/config_evoxbench.py`

#### Download and Install Mujoco
```bash
sudo apt update
sudo apt install g++
sudo apt-get upgrade libstdc++6
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libghc-x11-dev
sudo apt install libcairo2-dev pkg-config python3-dev
sudo apt-get install patchelf

wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz
mkdir ~/.mujoco
tar -zxvf mujoco210_linux.tar.gz -C ~/.mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

conda deactivate
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda activate gfmo

mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11
mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL
sudo cp /usr/include/X11/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11/
sudo cp /usr/include/GL/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL 
```

#### Run Experiments
```bash
conda activate gfmo
bash scripts/gfmo_all_exps.sh
```


## Use ParetoFlow as a pip package
### Installation

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

### Usage
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
all_x = np.load("examples/data/x.npy")
all_y = np.load("examples/data/y.npy")

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

### Examples
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
