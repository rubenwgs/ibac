#!/bin/bash
set -x # print commands

export method=dynerf_gaussian sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train tag=geometric$tag
