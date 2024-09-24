#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0,1

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file scripts/config/2gpu_g16.yaml \
    peft_finetune.py \
    --dataset_type pretrain \
    --peft_method splora \
    --dataset_name datasets/SlimPajama-0.5B-ap \
    --model_name_or_path checkpoints/Llama-2-7b-hf-sparse_2_4 \
    --lora_rank 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --block_size 2048 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/Llama-2-7b-hf-sparse_2_4-r16-0.5B-lr1e-3 \
    --num_warmup_steps 50

# --selected_indices_path checkpoints/Llama-2-7b-hf-sparse_2_4/selected_indices_c4.pth
