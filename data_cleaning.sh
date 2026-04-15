#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J "chexpert_prep"

module load python/3.11.6-gcc-13.2.0-fh6i4o3
python data_cleaning.py