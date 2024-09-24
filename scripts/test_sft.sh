#!/bin/bash
set -e
set -x

cd $(dirname $0)/..


bs=1
grad_acc=1
lora_rank=16
lr=1e-3
min_lr=1e-4

# meta-math, codefeedback, wizardlm

  # accelerate launch --config_file scripts/config/1gpu_g${grad_acc}.yaml \
  # --main_process_port $4 \
  # peft_finetune.py \

run_command () {
  CUDA_VISIBLE_DEVICES=$5 \
      python peft_finetune.py \
      --dataset_type sft-domain \
      --peft_method splora \
      --dataset_name $3 \
      --model_name_or_path $1 \
      --lora_rank ${lora_rank} \
      --per_device_train_batch_size ${bs} \
      --per_device_eval_batch_size 1 \
      --learning_rate ${lr} \
      --min_learning_rate ${min_lr} \
      --num_train_epochs 1 \
      --gradient_accumulation_steps ${grad_acc} \
      --report_to tensorboard \
      --with_tracking \
      --output_dir $2 \
      --num_warmup_steps 20
}

# run_command "checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "checkpoints/Llama-2-7b-hf_sparsegpt.2of4_lora-rank${lora_rank}_meta-math" "meta-math" 29550 "2"

# run_command "checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "checkpoints/Llama-2-7b-hf_sparsegpt.2of4_lora-rank${lora_rank}_codefeedback" "codefeedback" 29550 "2"

run_command "checkpoints/Llama-2-7b-hf_sparsegpt.2of4" "checkpoints/Llama-2-7b-hf_sparsegpt.2of4_lora-rank${lora_rank}_wizardlm" "wizardlm" 29550 "2"
