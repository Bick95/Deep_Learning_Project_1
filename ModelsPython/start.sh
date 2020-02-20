#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=RFA
#SBATCH --mail-type=END,FAILL
#SBATCH --mail-user=d.bick@student.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
python Resnet50_RMSP_Frozen_Aug.py
