#!/bin/bash
set -x

SEQUENCES=(basketball dancer exercise model)
METHODS=(dynerf_laplacian)

for sequence in "${SEQUENCES[@]}"; do
    for method in "${METHODS[@]}"; do
      tag="Laplacian_${sequence}"
      python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train tag=$tag
    done
done
