#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -J "dqy_exp1"
#SBATCH --output=/groups/CS156b/from_central/2026/JDP/dean_folder/j.out
#SBATCH --error=/groups/CS156b/from_central/2026/JDP/dean_folder/j.err

cd /groups/CS156b/from_central/2026/JDP/dean_folder
git pull 

# activate env
source ~/.bashrc
conda activate /groups/CS156b/from_central/2026/JDP/JDP-env

python3 exp1.py
