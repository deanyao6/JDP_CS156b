#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -J "frontal_resnet"
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

cd /groups/CS156b/from_central/2026/JDP/dean_folder
git pull 

# activate env
source ~/.bashrc
conda activate /groups/CS156b/from_central/2026/JDP/JDP-env

python3 exp1.py
