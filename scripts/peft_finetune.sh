#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

# datasets: 

devices=0

CUDA_VISIBLE_DEVICES=${devices} python peft_finetune.py \
    --dataset_type corpus \
    --peft_method lora_sparse \
    --dataset_name wikitext2 \
    --model_name_or_path checkpoints/Llama-2-7b-hf-wanda_2_4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --block_size 2048 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/tmp \
    --max_train_steps 1
