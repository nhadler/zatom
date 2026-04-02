#!/bin/bash -l

# First, get an interactive allocation with shifter:
#   salloc --nodes 4 --qos interactive --time 04:00:00 --constraint "gpu&hbm80g" \
#          --gpus-per-node 4 --ntasks-per-node 1 --account m636 \
#          --image=nersc/pytorch:25.02.01 --module=gpu,nccl-plugin
#
# Then run this script:
#   bash scripts/perlmutter/interactive/train_ddp_tft_tmqm.sh

PROJECT_DIR="/global/cfs/cdirs/m636/nhadler/projects/zatom"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR" || exit

# Define run details
RUN_DATE=$(date +%Y-%m-%d_%H-%M-%S)
TASK_NAME="train_fm"
RUN_NAME="train_tft_80M_tmqm"
CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}_${RUN_DATE}/checkpoints/"
mkdir -p "$CKPT_PATH"

# Inform user of job details
echo -e "Job details:\n========================================================================\n"

echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run start time: $RUN_DATE"

echo -e "\nSLURM job name: $SLURM_JOB_NAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM master node: $SLURMD_NODENAME"
echo "SLURM all nodes: $SLURM_NODELIST"
echo "SLURM node count: $SLURM_JOB_NUM_NODES"

echo -e "\nCUDA visible devices: $CUDA_VISIBLE_DEVICES"

echo -e "\nCurrent time: $(date)"
echo "Current directory: $(pwd)"
echo "Current node: $(hostname)"

echo -e "\nExecuting script $TASK_NAME.py:\n========================================================================\n"

# Set master address to first node in allocation
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Run script — each srun task sets GROUP_RANK so Lightning knows which node it's on
srun --kill-on-bad-exit=1 \
    --nodes=$SLURM_JOB_NUM_NODES \
    --ntasks-per-node=1 \
    --gpus-per-node=4 \
    shifter bash -c "
    export GROUP_RANK=\$SLURM_NODEID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    HYDRA_FULL_ERROR=1 $VENV_PYTHON zatom/$TASK_NAME.py \
        experiment=train \
        logger=tensorboard \
        environment=lightning \
        ckpt_path=$CKPT_PATH \
        date=$RUN_DATE \
        name=$RUN_NAME \
        task_name=$TASK_NAME \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        callbacks.model_checkpoint.monitor=val_tmqm/valid_rate \
        callbacks.model_checkpoint.every_n_epochs=5 \
        callbacks.model_checkpoint.save_top_k=10 \
        callbacks.last_model_checkpoint.every_n_epochs=5 \
        callbacks.last_model_checkpoint.every_n_train_steps=null \
        trainer.check_val_every_n_epoch=5 \
        data.datamodule.datasets.mp20.proportion=0.0 \
        data.datamodule.datasets.qm9.proportion=0.0 \
        data.datamodule.datasets.tmqm.proportion=1.0 \
        data.datamodule.batch_size.train=32 \
        data.datamodule.batch_size.val=32 \
        data.datamodule.batch_size.test=32 
"

echo "Training completed for SLURM job $SLURM_JOB_ID"
