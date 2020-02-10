#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=cnn-2
#SBATCH --mail-type=END,FAILL
#SBATCH --mail-user=s.jinde@student.rug.nl
#SBATCH --output=job-cnn-%j.log
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load Python/3.7.2-GCCcore-8.2.0
module load TensorFlow/2.0.0-foss-2019a-Python-3.7.2
python dl_resnet_50.py
