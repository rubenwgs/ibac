#!/bin/bash
#SBATCH --chdir .
#SBATCH --account digital_humans
#SBATCH --time=12:00:00
#SBATCH -o /home/%u/slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=14G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate Digital_Humans_GPU

export PYTHONPATH="${PYTHONPATH}:/cluster/courses/digital_humans/datasets/team_4/balfourb/PycharmProjects/Digital_Humans_Shared/"
python src/train.py

echo "Done."
echo FINISHED at $(date)