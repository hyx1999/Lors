#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

bs=1
grad_acc=1
lora_rank=16
lr=1e-3
min_lr=1e-4

run_command () {
  # CUDA_VISIBLE_DEVICES=0 \
      # accelerate launch --config_file scripts/config/1gpu_g1.yaml \
  CUDA_VISIBLE_DEVICES=0 python peft_finetune.py \
      --dataset_type pretrain \
      --peft_method $2 \
      --dataset_name datasets/SlimPajama-0.5B-ap \
      --model_name_or_path $1 \
      --lora_rank ${lora_rank} \
      --per_device_train_batch_size ${bs} \
      --per_device_eval_batch_size 1 \
      --learning_rate ${lr} \
      --min_learning_rate ${min_lr} \
      --num_train_epochs 1 \
      --gradient_accumulation_steps ${grad_acc} \
      --block_size 2048 \
      --num_warmup_steps 20 \
      --max_train_steps 64 \
      --no_test
}

# run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "lora"

# run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "splora"

# run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "splora-gradckpt"

# run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "splora-naive"

# run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "spp"

run_command "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "spp-naive"
