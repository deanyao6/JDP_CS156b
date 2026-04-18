#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -J "data_cleaning"
#SBATCH --mail-user=jwang8@caltech.edu 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /resnick/groups/CS156b/from_central/2026/JDP/jenna

source ~/miniconda3/bin/activate cs156b
python data_cleaning.py