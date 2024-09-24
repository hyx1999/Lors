#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0,3
grad_acc=32

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file scripts/config/2gpu_g${grad_acc}.yaml \
    peft_finetune.py \
    --dataset_type sft \
    --peft_method spp \
    --dataset_name alpaca \
    --model_name_or_path checkpoints/Llama-2-7b-hf_wanda.2of4 \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps ${grad_acc} \
    --block_size 2048 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/Llama-2-7b-hf_wanda.2of4_lora-rank16_alpaca_lr1e-3_spp

# --selected_indices_path checkpoints/Llama-2-7b-hf-sparse_2_4/selected_indices_c4.pth
