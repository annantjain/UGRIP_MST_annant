#!/bin/bash

#SBATCH --job-name=0610mbert1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=it-hpc
#SBATCH --reservation=ugrip-nlp
#SBATCH --output=0610mbert1.out
#SBATCH --error=0610mbert1.err

python statement_tune.py --Exp_name "MBERT" --transformer "google-bert/bert-base-multilingual-cased" --tr_ep 6 --save "./MBERT2" --tr_batch 32 --ev_batch 32