#!/bin/bash
set -x

SEQUENCES=(basketball dancer exercise model)
METHODS=(tnerf tnerf_gaussian)

for sequence in "${SEQUENCES[@]}"; do
    for method in "${METHODS[@]}"; do

      python launch.py                        \
      --config ./configs/dysdf/$method.yaml   \
      dataset.scene=$sequence                 \
      --exp_dir ../exp_owlii_benchmark        \
      --train                                 \
      tag="${method}_${sequence}"             \

    done
done
