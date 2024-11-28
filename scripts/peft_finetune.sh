#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

# datasets: 

devices=0

# /data/slimpajama-0.5B-Llama-3-tokenized
CUDA_VISIBLE_DEVICES=${devices} python peft_pretrain.py \
    --peft_method lors \
    --dataset_name wikitext2 \
    --model_name_or_path checkpoints/TinyLlama-1.1B-Chat \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --block_size 2048 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/tmp


CUDA_VISIBLE_DEVICES=${devices} python peft_sft.py \
    --peft_method lors \
    --dataset_name alpaca \
    --model_name_or_path checkpoints/TinyLlama-1.1B-Chat \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --cutoff_len 2048 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/tmp


# meta-math, wizardlm, codefeedback
CUDA_VISIBLE_DEVICES=${devices} python peft_domain.py \
    --peft_method lors \
    --dataset_name meta-math \
    --model_name_or_path checkpoints/TinyLlama-1.1B-Chat \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-4 \
    --cutoff_len 2048 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir checkpoints/tmp
