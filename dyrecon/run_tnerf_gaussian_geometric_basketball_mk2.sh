#!/bin/bash
set -x # print commands

export method=tnerf_gaussian_geometric_mk2 sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train tag=tnerf_gaussian_geometric_6_layers_with_skip_mk2$tag