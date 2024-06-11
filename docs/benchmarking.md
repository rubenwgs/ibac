# Benchmarking

This document contains instructions on how reproduce results from the project.

## 2D video approximation

To reproduce results on the video approximation benchmark from are report you may execute the bash script:
```
bash run_video_benchmark.sh
```


Otherwise you can adapt the following bash script to select a different `video_path` and `model` config.
```bash

#### CHOOSE DATASET SEQUENCE ####

export sequence="../DATA_ROOT/video/cat.mp4" 
# export squence="skvideo.datasets.bikes" 

#### CHOOSE MODEL  ####

export model=".configs/video/relu_base.yaml"
# export model=".configs/video/siren_base.yaml"
# export model=".configs/video/gaussain_base.yaml"
# export model=".configs/video/laplacian_base.yaml"
# export model=".configs/video/expsin_base.yaml"
# export model=".configs/video/quadratic_base.yaml"


### RUN VANILLA MLP ###

python launch.py                                        \
    --config $model                                     \
    --train                                             \
    --predict                                           \
    dataset.video_path=$sequence                        \
    --exp_dir ../exp_video                              \
    model.hidden_features=512                           

### RUN RESFIELD MLP ###

python launch.py                                        \
    --config $model                                     \
    --train                                             \
    --predict                                           \
    dataset.video_path=$sequence                        \
    --exp_dir ../exp_video                              \
    model.hidden_features=512                           \
    # Add resfield parameters + tag
    model.resfield_layers=[1,2,3]                       \
    model.composition_rank=40                           \
    tag=ResFields 

```

Note that a new and unique `expdir` must be given every run otherwise the files may be overwritten or lost. The `tag` argument gets appended to the output directory of the run. 


## Dynamic NeRF from 4 RGB views

The code implements the  following nerf models 

1. [TNeRF](https://neural-3d-video.github.io/) Li et al. (CVPR 2022) and Pumarola et al. (CVPR 2021)
2. [DyNeRF](https://neural-3d-video.github.io/) Li et al., CVPR 2022
3. [DNeRF](https://neural-3d-video.github.io/) Pumarola et al., CVPR 2021
4. [Nerfies](https://github.com/google/nerfies), Park et al., ICCV 2021
5. [HyperNeRF](https://github.com/google/hypernerf), Park et al., SIGGRAPH Asia 2021
6. [NDR](https://github.com/USTC3DV/NDR-code), Cai et al., NeurIPS 2022

As previously mentioned we only derived geometric intialization for the gaussian activation function. Training a nerf model with any other of the proposed activations will likely result in the output to be an empty mesh.

Due to time constaints we only tested the for [TNeRF](https://neural-3d-video.github.io/) Li et al. (CVPR 2022) and Pumarola et al. (CVPR 2021) variant (as per reccomendation from our superviser).

From our understanding, however, there is no reason why the provided model implementations shouldn't work (for gaussian activation).
The only reason why they were not tested is due to limitations in time and 
resources.

Similar to the video task you may run the following script to run the entire benchmark 

```

```

Otherwise you can adapt the following bash script to select a different `dataset` and `sdf_network`.





