#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J "chexpert_prep"

cd /resnick/groups/CS156b/from_central/2026/JDP/jenna
git pull origin main

source ~/miniconda3/bin/activate cs156b
python data_cleaning.py