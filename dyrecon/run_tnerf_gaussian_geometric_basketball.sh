#!/bin/bash
set -x # print commands

export method=tnerf_gaussian_geometric sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train tag=tnerf_gaussian_geometric_4_layers$tag