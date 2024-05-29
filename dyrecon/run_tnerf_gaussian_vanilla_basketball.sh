#!/bin/bash
set -x # print commands

export method=tnerf_gaussian_vanilla sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train tag=t_nerf_gaussian_vanilla$tag