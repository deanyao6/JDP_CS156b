#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J "chexpert_prep"

source ~/miniconda3/bin/activate cs156b
python data_cleaning.py