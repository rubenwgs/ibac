# Installation

## Conda Environment 

Note that the conda environment requirements are different from those of the paper. I'm using a more updated version. 
You should set up a conda environment as follows.

For mac you should use the following 
```
conda create -n Digital_Humans_MAC pytorch::pytorch torchvision torchaudio torchmetrics lightning "python>=3.10" hydra-core rich -c conda-forge -c pytorch
conda activate Digital_Humans_MAC
pip install hydra-optuna-sweeper hydra-colorlog
```

For a pc with nvidia gpu instead use the following

```
conda create -n Digital_Humans_GPU "python>=3.10" pytorch torchvision torchaudio pytorch-cuda=12.1 torchmetrics lightning hydra-core rich  -c conda-forge -c pytorch -c nvidia
conda activate Digital_Humans_GPU
pip install hydra-optuna-sweeper hydra-colorlog
```

## Dataset

you should install the dataset from the resfields github repository here as specified [here](https://github.com/markomih/ResFields/blob/master/docs/data.md)


