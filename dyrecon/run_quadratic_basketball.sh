#!/bin/bash
set -x # print commands

export method=dynerf_quadratic sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train
