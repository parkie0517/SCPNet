# SCPNet
This repo is based on the official SCPNet code.

## 1. Environment Setup Guide (Works as of 2024-04-02)
- python==3.7
    - conda create -n SCPNet python=3.7
    - conda activate SCPNet
    - conda config --set ssl_verify false
- torch 1.10.0, cuda==11.3 (if you get a Conda HTTP error, keep on reinstalling until it finishes. There is no possible solution for this as this is a network issue)
    - conda install -y pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
- check if CUDA is available
    - CUDA_VISIBLE_DEVICES=1 python
    - import torch
    - print(torch.cuda.is_available())
- pyyaml
    - conda install -y anaconda::pyyaml
- Cython
    - conda install -y anaconda::cython
- tqdm
    - conda install -y anaconda::tqdm
- numba
    - conda install -y anaconda::numba
- Numpy-indexed
    - conda install -y conda-forge::numpy-indexed
- torch-scatter (takes some time for this to install)
    - conda install -y pytorch-scatter -c pyg
- spconv==1.0
    - git clone https://github.com/tyjiang1997/spconv1.0.git  --recursive
    - sudo apt-get install libboost-all-dev
    - conda install -y anaconda::cmake
    - 
    - 
- strictyaml
    - pip install strictyaml

## 2. Dataset Preparation


## 3. Error Fix to the Original Code
1. AttributeError in "segmentator_3d_asymm_spconv.py"  
Used "from spconv.pytorch.conv import SubMConv3d, SparseConv3d, SparseInverseConv3d" in line 7

## 4. Things I Was Curious About
- How does SCPNet address the problem of information loss in the segmentation sub-network?
    - By using an MPB (multi-path block) instead of pooling operations.
    - MPB is composed of 3D convolution operations.
 