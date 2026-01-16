#!/bin/bash
#SBATCH --job-name=qmof150_only_tft80m
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=32G
#SBATCH --time 48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:2
#SBATCH --output=logs/train_qmof150_only_tft80m_h200x2_%j.out
#SBATCH --error=logs/train_qmof150_only_tft80m_h200x2_%j.err

# Change to the repo root directory
cd /gpfs/radev/pi/ying_rex/afp38/zatom

# Set checkpoint directory to GPFS location with sufficient space
CHECKPOINT_DIR="/gpfs/radev/pi/ying_rex/afp38/zatom_checkpoints_qmof150_only"
mkdir -p "${CHECKPOINT_DIR}"

# Set WandB cache to GPFS to avoid home directory quota issues
export WANDB_CACHE_DIR="${CHECKPOINT_DIR}/wandb_cache"
export WANDB_DIR="${CHECKPOINT_DIR}/wandb_runs"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${WANDB_DIR}"

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export CUDA_LAUNCH_BLOCKING=0
export WANDB_RESUME=allow
export WANDB_RUN_ID=62yoj3sx

# Train QMOF150-only TFT-80M model
python zatom/train_fm.py \
  experiment=train_qmof150_only \
  data.datamodule.batch_size.train=128 \
  ckpt_path=/gpfs/radev/pi/ying_rex/afp38/zatom_checkpoints_qmof150_only/checkpoints/last.ckpt \
  resume_from_last_step_dir=false \
  logger.wandb.entity=zatom \
  logger.wandb.project=zatom \
  logger.wandb.tags=[qmof150_only,tft80m,generation] \
  name="train_qmof150_only_tft80m_h100x2" \
  callbacks.last_model_checkpoint.dirpath="${CHECKPOINT_DIR}/checkpoints" \
  callbacks.model_checkpoint.dirpath="${CHECKPOINT_DIR}/checkpoints" \
  trainer.default_root_dir="${CHECKPOINT_DIR}" \
  paths.viz_dir="${CHECKPOINT_DIR}/visualizations" \
  logger.wandb.id=62yoj3sx \
  model.sampling.save_dir="${CHECKPOINT_DIR}/visualizations" \
  model.sampling.save=true \
  model.sampling.visualize=false


