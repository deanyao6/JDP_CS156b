#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J "resnet_lr1e3_epoch10_predict"
#SBATCH --output=resnet_lr1e3_epoch10_predict_%j.out
#SBATCH --error=resnet_lr1e3_epoch10_predict_%j.err
#SBATCH --mail-user=jwang8@caltech.edu 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /resnick/groups/CS156b/from_central/2026/JDP/jenna
source ~/miniconda3/bin/activate cs156b
python resnet_frontal_predict.py \
    --checkpoint checkpoints/resnet_lr1e3/best_resnet50.pth \
    --output submission_resnet_lr1e3_epoch10.csv