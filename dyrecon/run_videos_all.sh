#!/bin/bash
set -x

SEQUENCES=(../DATA_ROOT/Video/cat.mp4 skvideo.datasets.bikes)
MODELS=(relu_base siren_base gaussian_base quadratic_base laplacian_base gaussian_base)


for sequence in "${SEQUENCES[@]}"; do
    for model in "${MODELS[@]}"; do

      # RUN VANILLA MODEL
      python launch.py                                        \
          --config "./configs/video/${model}.yaml"             \
          --train                                             \
          --predict                                           \
          dataset.video_path="${sequence}"                    \
          --exp_dir ../exp_video                              \
          model.hidden_features=512                           \
          tag="Resfields_${model}"
      #RUN RESFIELD MODEL
      python launch.py                                        \
          --config "./configs/video/${model}.yaml"             \
          --train                                             \
          --predict                                           \
          dataset.video_path="${sequence}"                    \
          --exp_dir ../exp_video                              \
          model.hidden_features=512                           \
          model.resfield_layers=[1,2,3]                       \
          model.composition_rank=40                           \
          tag="Resfields_${model}"

    done
done
