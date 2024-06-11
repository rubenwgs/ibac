# Installation 

**NOTE** This page was adapted that of the [resfields paper](https://markomih.github.io/ResFields/). 

The code was tested on Ubuntu 22.04 with CUDA 11.6 and Python 3.9.
The experiments were run on the ethz euler gpu clusters using rtx 3090 gpus

## 1. Clone the repo

```bash
https://github.com/rubenwgs/ibac.git
cd ibac
```

## 2. Create and Install Environment

We create a new `conda` environment in [Anaconda](https://www.anaconda.com/)

```bash 
conda create -n Digital_Humans_Project python=3.9 -y
conda activate Digital_Humans_Project
```
Install the nvidia `cudatoolkit` version 11.6 from the nvdia channel

```bash
conda install conda install cudatoolkit=11.6 -c nvidia  -y
```

Install `pytorch` compatible with the `cudatoolkit` using the `pip` inside the conda environment. (Make sure that the pip command refers to that of the newly created conda envifronment and not that of the system)

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

then we install all remaining dependencies using `pip` (same as with pytorch).

```bash
pip install\
    tqdm \
    scikit-image \
    opencv-python \
    configargparse \
    lpips \
    imageio-ffmpeg \
    lpips \
    tensorboard \
    numpy==1.22.4 \
    sk-video\
    trimesh \
    wandb \
    omegaconf \
    pysdf \
    pymcubes \
    matplotlib \
    pytorch-lightning==1.6.5 \
    gdown
```


## 3. [optional] Download the data

see [data preparation](/docs/data.md) to set up the datasets