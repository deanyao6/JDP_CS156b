#!/bin/bash
#SBATCH --account=cs156b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=densenet_DY
#SBATCH --output=/resnick/groups/CS156b/from_central/2026/JDP/dean/densenet_DY_%j.out
#SBATCH --mail-user=dyao2@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# activate your conda env — run `conda env list` on HPC to find the right name
source ~/.bashrc
conda activate /groups/CS156b/from_central/2026/JDP/JDP-env

cd /resnick/groups/CS156b/from_central/2026/JDP

python densenet121.py
