#!/bin/bash --login

#SBATCH --time 03-00:00:00
#SBATCH --output logs/dynamics.log
#SBATCH --open-mode append
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 6
#SBATCH --mem 128G

source ~/.wandb_api_key

date +%F:%H-%M-%S
srun python main.py \
    --base configs/surg_transfer.yaml \
    --logdir /scratch/dthapa/logs/surg_transfer \
    2>&1 | tee output_train.log

printf "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
