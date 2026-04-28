#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=cs156b
#SBATCH -J "densenet_DY"
#SBATCH --output=densenet_DY_%j.out
#SBATCH --error=densenet_DY_%j.err
#SBATCH --mail-user=dyao2@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /resnick/groups/CS156b/from_central/2026/JDP/dean_folder
git pull origin main

source ~/miniconda3/bin/activate /groups/CS156b/from_central/2026/JDP/JDP-env

python densenet121.py
