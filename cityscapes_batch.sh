#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --array=[1-10]
#SBATCH -J from_cityscapes_to_acdc
#SBATCH -o slurm_logs/%x.%J.out
#SBATCH -e slurm_logs/%x.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=malfarra@intel.com
#SBATCH --mail-type=ALL

source activate test_env

PORT=$(shuf -i 33000-65000 -n 1)

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
run.py --num_workers 2 --dataset cityscapes_domain --data_root /export/share/datasets/cityscapes/ \
--method FT --num_classes 19 --epochs 1 --number_internal_steps ${SLURM_ARRAY_TASK_ID} \
--name ${SLURM_ARRAY_TASK_ID}_steps_1_epoch_cityscapes_to_acdc_cityscapes  --task offline


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
run.py --num_workers 2 --dataset acdc --data_root /home/malfarra/acdc \
--method FT --num_classes 19 --epochs 1 --number_internal_steps 0 --lr 0 \
--name ${SLURM_ARRAY_TASK_ID}_steps_1_epoch_cityscapes_to_acdc_acdc  --task offline \
--test --ckpt ./checkpoints/step/${SLURM_ARRAY_TASK_ID}_steps_1_epoch_cityscapes_to_acdc_cityscapes

