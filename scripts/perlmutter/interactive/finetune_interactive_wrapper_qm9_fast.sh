#!/bin/bash

MAX_RETRIES=120
COUNT=1

while [ $COUNT -le $MAX_RETRIES ]; do
    echo "[$COUNT/$MAX_RETRIES] Requesting new interactive allocation..."

    salloc -C "gpu&hbm80g" \
           --qos=interactive \
           --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 \
           --module=gpu,nccl-plugin \
           --account=m5008 \
           --nodes=2 \
           --gpus-per-node=2 \
           --ntasks-per-node=2 \
           --time=04:00:00 \
           --job-name=finetune-tft-70M-qm9-fast \
           bash -c "bash scripts/perlmutter/interactive/finetune_ddp_tft_qm9_fast.sh"

    echo "Job finished or timed out. Restarting..."
    COUNT=$((COUNT + 1))
    sleep 10  # optional pause before retry
done

echo "Reached maximum number of retries ($MAX_RETRIES). Stopping."
