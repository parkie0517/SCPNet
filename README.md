# SCPNet
This repo is based on the official SCPNet code.

## 1. Environment Setup Guide
- python==3.7
    - conda create -n SCPNet python=3.7
    - conda activate SCPNet
- cuda==11.3
    - conda config --set ssl_verify no
    - conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
- pyyaml
    - pip install pyyaml
- Cython
    - pip install Cython
- tqdm
    - pip install tqdm
- numba
    - pip install numba
- Numpy-indexed
    - pip install numpy-indexed
- torch-scatter
    - conda install pytorch-scatter -c pyg
- spconv
    - pip install spconv-cu113
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