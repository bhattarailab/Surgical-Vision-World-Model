#!/bin/bash --login

#SBATCH --time 03-00:00:00
#SBATCH --output logs/action.log
#SBATCH --open-mode append
#SBATCH --gres gpu:2
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 8
#SBATCH --mem 48G

TAG=surg_transfer_lam
NUM_WORKERS=8

source ../.venv/bin/activate
source ~/.wandb_api_key # export your wandb api key here
export WANDB_DIR=/scratch/dthapa/wandb

accelerate launch --multi_gpu --main_process_port 0 train_action.py \
--config config/surg_transfer.yaml \
--log_every 1 \
--num_workers ${NUM_WORKERS} \
--tag ${TAG} \

printf "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
