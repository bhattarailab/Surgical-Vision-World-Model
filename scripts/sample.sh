#!/bin/bash --login

#SBATCH --time 12:00:00
#SBATCH --output logs/sample.log
#SBATCH --open-mode append
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 400G

python sample.py \
	--ckpt /scratch/dthapa/checkpoints/surg_transfer_dynamics/pytorch_model.bin \
	--config_path configs/surg_transfer.yaml \
	--save_folder /scratch/dthapa/outputs/test \
	--batch_size 48 \
	--T 7 \
	--T_sample 14 \
	--num_videos 1000 \
	--compute_metrics

printf "\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
