#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J "frontal_resnet_full"
#SBATCH --output=resnet_frontal_predict_full_%j.out
#SBATCH --error=resnet_frontal_predict_full_%j.err
#SBATCH --mail-user=jwang8@caltech.edu 
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /resnick/groups/CS156b/from_central/2026/JDP/jenna
source ~/miniconda3/bin/activate cs156b
python resnet_frontal_predict.py \
    --checkpoint checkpoints/resnet_frontal_full_h200/best_resnet50.pth \
    --output submission_resnet_frontal_full_h200.csv