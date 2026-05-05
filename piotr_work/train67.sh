#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --account=cs156b
#SBATCH -J train67_pz
#SBATCH --output=train67_pz_%j.out
#SBATCH --error=train67_pz_%j.err
#SBATCH --mail-user=pzawisla@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /resnick/groups/CS156b/from_central/2026/JDP/piotr_repo

git pull

cd piotr_work

mkdir -p logs

/resnick/groups/CS156b/from_central/2026/JDP/JDP-env/bin/python3 train67.py