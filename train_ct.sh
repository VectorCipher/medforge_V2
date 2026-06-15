#!/bin/bash

# Configuration
# Path to the Kaggle MedForge-Reasoner checkpoint directory
MODEL_CHECKPOINT="/kaggle/working/MedForge-Reasoner"
DATASET_PATH="/kaggle/working/sft_train_ct.json" # Assuming output of prepare_dataset.py goes here or you run it in working dir
OUTPUT_DIR="/kaggle/working/output/medforge-ct-finetune"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting Fine-Tuning on CT Lungs dataset..."

# Environment variables for distributed training
export MASTER_PORT=29502
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0  # Change to 0,1,2,3 if you have multiple GPUs
export NPROC_PER_NODE=1        # Change this to match the number of GPUs

# Run swift SFT
# Note: LoRA alpha/rank and learning rates are optimized based on MedForge's original training
swift sft \
    --model "$MODEL_CHECKPOINT" \
    --model_type qwen3_vl_8b_instruct \
    --dataset "$DATASET_PATH" \
    --split_dataset_ratio 0.05 \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --padding_free true \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --report_to tensorboard
