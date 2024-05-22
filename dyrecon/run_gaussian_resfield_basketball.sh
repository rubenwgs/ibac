#!/bin/bash
set -x # print commands

export method=dynerf_gaussian sequence=basketball

python launch.py --config ./configs/dysdf/$method.yaml dataset.scene=$sequence --exp_dir ../exp_owlii_benchmark --train  model.sdf_net.resfield_layers=[1,2,3,4,5,6,7] tag=ResFields1234567
