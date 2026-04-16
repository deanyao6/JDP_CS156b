#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -J "val_densenet121"
#SBATCH --output=val_%j.out
#SBATCH --error=val_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dyao2@caltech.edu

cd /resnick/groups/CS156b/from_central/2026/JDP/dean_folder

source ~/.bashrc
conda activate /resnick/groups/CS156b/from_central/2026/JDP/JDP-env

python3 val_dn121.py