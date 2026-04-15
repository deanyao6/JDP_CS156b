#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -J "frontal_resnet"
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

cd ~/JDP_CS156b
git pull origin cs156b_branch

source ~/miniconda3/bin/activate cs156b

python resnet_frontal_training.py \
    --train_csv train_clean.csv \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 4 \
    --output_dir checkpoints