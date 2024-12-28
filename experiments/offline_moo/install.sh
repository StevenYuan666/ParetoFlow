#!/bin/bash
YOUR_PATH_TO_CONDA=~/anaconda3

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

conda env create -f environment.yml
conda activate off-moo
# conda create -n off-moo python=3.8
# conda activate off-moo
conda install gxx_linux-64 gcc_linux-64
conda install --channel=conda-forge libxcrypt
pip install -r requirements.txt
pip install torchvision==0.11.1
# https://download.pytorch.org/whl/cu117 if you are using CUDA 11.7
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
cd off_moo_bench/problem/lambo
pip install -e .
cd ../../../
pip install scipy==1.10.1
pip install scikit-learn==0.21.3
pip install --upgrade pandas
pip install --upgrade kiwisolver

# Sklearn==0.21.3 might be wrong in some scenarios, thus we fix bugs with the scripts below.
# Please set your path to conda here
bash fix_contents.sh ${YOUR_PATH_TO_CONDA}/envs/off-moo/lib/python3.8/site-packages/sklearn/cross_decomposition/pls_.py "pinv2" "pinv"

# Make sure that you have paste your Google Drive API below
curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/11bQ1paHEWHDnnTPtxs2OyVY_Re-38DiO/view?usp=sharing -o database.zip
curl -H "Authorization: Bearer <Your Google Drive APIs>" https://drive.google.com/file/d/1r0iSCq1gLFs5xnmp1MDiqcqxNcY5q6Hp/view?usp=sharing -o data.zip

unzip database.zip -d off_moo_bench/problems/mo_nas/
unzip data.zip -d off_moo_bench/problems/mo_nas/

conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda activate off-moo

cd off_moo_bench/problem/lambo
# It will take some time
python scripts/black_box_opt.py optimizer=mf_genetic optimizer/algorithm=nsga2 task=proxy_rfp tokenizer=protein 
cd ../../../

# For MuJoCo, if it raises an error that .h file is not found, an easy way is to copy that from /usr/include
mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11
mkdir ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL
sudo cp /usr/include/X11/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/X11/
sudo cp /usr/include/GL/*.h ${YOUR_PATH_TO_CONDA}/envs/off-moo/include/GL

python config_evoxbench.py 
python tests/test_mujoco.py
python tests/test_env.py
python tests/test_vallina_modules.py



