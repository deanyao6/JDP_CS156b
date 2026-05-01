#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -J "frontal_resnet_full_2"
#SBATCH --mail-user=jwang8@caltech.edu 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

cd /resnick/groups/CS156b/from_central/2026/JDP/jenna
git pull origin main

source ~/miniconda3/bin/activate cs156b

python resnet_frontal_training_2.py \
    --train_csv train_clean.csv \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 0 \
    --output_dir checkpoints