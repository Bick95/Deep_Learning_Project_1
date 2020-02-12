#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=finetuning
#SBATCH --mail-type=END,FAILL
#SBATCH --mail-user=s.jinde@student.rug.nl
#SBATCH --output=job-cnn-%j.log
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
python dl_inception_finetuning.py
