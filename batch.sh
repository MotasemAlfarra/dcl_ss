#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=[1-10]
#SBATCH -J test_run_to_visualize_outputs
#SBATCH -o slurm_logs/output_no_bg.%J.out
#SBATCH -e slurm_logs/error_no_bg.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=malfarra@intel.com
#SBATCH --mail-type=ALL

source activate test_env

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$(shuf -i 33000-65000 -n 1) \
run.py --num_workers 2 --dataset cityscapes_domain --data_root /export/share/datasets/cityscapes/ \
--method FT --num_classes 19 --epochs 1 --number_internal_steps ${SLURM_ARRAY_TASK_ID}  \
--name ${SLURM_ARRAY_TASK_ID}_steps_1_epoch_default_arguments  --task offline

