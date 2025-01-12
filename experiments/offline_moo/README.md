# Offline Multi-Objective Optimization

Benchmark and baselines for offline multi-objective optimization.



## Benchmark Installation

For a stable installation and usage, we suggest that you use a machine with ``CUDA version 11.7`` or higher.

### Data Downloading

Our proposed offline collected data can be accessed and downloaded via [Google Drive](https://drive.google.com/drive/folders/1SvU-p4Q5KAjPlHrDJ0VGiU2Te_v9g3rT?usp=drive_link).

### FoldX

In order to run our Regex, RFP, and ZINC tasks, following LaMBO ([Paper](https://arxiv.org/abs/2203.12742), [Code](https://github.com/samuelstanton/lambo)), you may first download [FoldX](https://foldxsuite.crg.eu/academic-license-info) Emulator.

[FoldX](https://foldxsuite.crg.eu/academic-license-info) is available under a free academic license. After creating an account you will be emailed a link to download the FoldX executable and supporting assets. Copy the contents of the downloaded archive to ``~/foldx``. You may also need to rename the FoldX executable (e.g. ``mv -v ~/foldx/foldx_20221231 ~/foldx/foldx``).

After installing FoldX, generate an instance ``proxy_rfp_problem.pkl`` of RFP task by running
```shell
cd off_moo_bench/problem/lambo/
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=proxy_rfp tokenizer=protein
```

Make sure that the lines of saving instance of ``proxy_rfp_problem.pkl`` exist in line 203 of  ``off_moo_bench/problem/lambo/lambo/optimizers/pymoo.py`` such that 
```python
if round_idx == self.num_rounds:
    import pickle
    with open('proxy_rfp_problem.pkl', 'wb+') as f:
        pickle.dump(problem, f)
```

### Quick Installation

After successfully installing [FoldX](https://foldxsuite.crg.eu/academic-license-info), run 
```shell
bash install.sh
```
for a quick installation.

### EvoXBench

To test with our offline MO-NAS problems, installing EvoXBench ([paper](https://arxiv.org/abs/2208.04321), [code](https://github.com/EMI-Group/evoxbench)) is needed. Before running ``pip install evoxbench``, you should first download their database via [Google Drive](https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1PwWloA543-81O-GFkA7GKg), and data via [Google Drive](https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/17dUpiIosSCZoSgKXwSBlVg), and save it to the path ``off_moo_bench/problem/mo_nas/database`` and ``off_moo_bench/problem/mo_nas/data``.
>  Following [How to download from Google Drive](https://www.quora.com/How-do-I-download-a-very-large-file-from-Google-Drive/answer/Shane-F-Carr), we propose a more stable method for downloading in a commandline interface. Please first go to [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/) to obtain your Google Drive download APIs, then download it with
> ```shell
> curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing -o database.zip
> curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view?usp=sharing -o data.zip
> ```
Then configure the EvoXBench as:
```python
    from evoxbench.database.init import config

    config("Path to database", "Path to data")
    # For instance:
    # With this structure:
    # /home/Downloads/
    # └─ database/
    # |  |  __init__.py
    # |  |  db.sqlite3
    # |  |  ...
    # |
    # └─ data/
    #    └─ darts/
    #    └─ mnv3/
    #    └─ ...
    # Then, execute:
    # config("PATH_TO_M2BO/off_moo_bench/problem/mo_nas/database", "PATH_TO_M2BO/off_moo_bench/problem/mo_nas/data")
```

### MuJoCo 
1. We use [MuJoCo](https://github.com/google-deepmind/mujoco) with version of ``2.1.0``, which can be downloaded on [GitHub](https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz), then run
```shell
mkdir ~/.mujoco
tar -zxvf mujoco210_linux_x86_64.tar.gz -C ~/.mujoco
```
to put MuJoCo under ``~/.mujoco`` and set environment variables. 

2. Make sure that you have installed needed dependancy.
```shell
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

3. Install required packages ``cython``, ``mujoco_py``, ``gym`` and set environmental variables.
```shell
pip install cython==3.0.0a10 mujoco-py==2.1.2.14 gym==0.14.0
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

4. After the above steps, you can check MuJoCo installation by 
```python
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]


sim.step()
print(sim.data.qpos)
# [-1.12164337e-05  7.29847036e-22  1.39975300e+00  9.99999999e-01
#   1.80085466e-21  4.45933954e-05 -2.70143345e-20  1.30126513e-19
#  -4.63561234e-05 -1.88020744e-20 -2.24492958e-06  4.79357124e-05
#  -6.38208396e-04 -1.61130312e-03 -1.37554006e-03  5.54173825e-05
#  -2.24492958e-06  4.79357124e-05 -6.38208396e-04 -1.61130312e-03
#  -1.37554006e-03 -5.54173825e-05 -5.73572648e-05  7.63833991e-05
#  -2.12765194e-05  5.73572648e-05 -7.63833991e-05 -2.12765194e-05]
```


### Notice during installation

1. Due to conflict versions among different packages, we design special orders to install packages correctly. Note that conducting ``conda env create -f environment.yml`` may raise some errors during installing Pip dependancy. Do not stop then and keep going on with rest scripts we proposed in ``install.sh``.
2. Due to packages conflicts, we use ``fix_contents.sh`` to solve such conflict bugs by running
    ```shell
    bash fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/off-moo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"
    ```
3. We provide serveral test suites under ``tests/`` folder to check for successful installation.
4. If you meet up with ``libstdc++.so.6: version `GLIBCXX_3.4.29' not found`` issue, since the reasons can be various, we recommend that you refer to [StackOverflow: Where can I find GLIBCXX_3.4.29?](https://stackoverflow.com/questions/65349875/where-can-i-find-glibcxx-3-4-29) or [GitHub: libstdc++.so.6: version `GLIBCXX_3.4.29' not found](https://github.com/pybind/pybind11/discussions/3453) for further advice to solve this issue.
5. We have test different hardware environments, including:
    - Ubuntu 22.04, 4x4090, CUDA 12.3
    - Ubuntu 22.04, 4x4090, CUDA 12.1
    - Ubuntu 22.04, 8xV100(32G), CUDA 12.1
    - Ubuntu 22.04, 1x3090, CUDA 12.0
    - Ubuntu 20.04, 2x3090, CUDA 11.8
    - Ubuntu 20.04, 2xA6000, CUDA 11.8



## Baselines

To reproduce the performance of baseline algorithms reported in our work, you may then run ``end2end.sh/multi_head.sh/multiple_models.sh/mobo.sh``, or run the following series of commands in a bash terminal. Also, please ensure that the conda environment ``off-moo`` is activated in the bash session.

> If you want to change some configurations, fix it in the script or in ``./config/``.

<!-- ### Notes for baselines
- For running special models (``GradNorm``, ``COMs``, etc), data pruning is needed, which corresponds to ``onlybest_1`` in ``--train-data-mode``.
- If you want to implement your own algorithms, put ``model`` and ``trainer`` under ``algorithm`` folder, then choose one MO-solver as what is done in ``scripts/multi_obj_nn.py``. -->

## Contact 
If you have any questions, feel free to contact [Rongxi Tan](https://trxcc.github.io/) or raise an issue.