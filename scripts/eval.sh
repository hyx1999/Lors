#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=3

CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task human-eval \
    --model /data2/models/hyx/Llama-3.1-8b-inst-hf_sparsegpt.2of4_lora-rank128_5B_alpaca_codefeedback \
    --model_name Llama-3.1-8b-inst-hf_sparsegpt.2of4_lora-rank128_5B_alpaca_codefeedback


# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model /data2/meta-llama/Meta-Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct

# lm_eval --model hf \
#     --model_args pretrained=/fs/fast/u2021000902/hyx/Llama-3.1-8b-inst-hf_sparsegpt.2of4_lora-rank128_5B_alpaca_meta-math \
#     --tasks gsm8k \
#     --device cuda:0 \
#     --batch_size auto \
#     --num_fewshot 0

# lm_eval --model hf \
#     --model_args pretrained=/fs/fast/u2021000902/hyx/Llama-3.1-8b-inst-hf_meta-math \
#     --tasks gsm8k \
#     --device cuda:0 \
#     --batch_size auto \
#     --num_fewshot 0

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-3-8b-hf

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf_wanda.2of4_lora-rank16_alpaca_lr1e-3_spp

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model /data2/models/hyx/Llama-3-8b-hf_wanda.2of4_lora-rank16_alpaca_lr1e-3

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model /data2/models/hyx/Llama-3-8b-hf_sparsegpt.2of4_lora-rank16_alpaca_lr1e-3

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-13b-hf

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-13b-hf_wanda.2of4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-13b-hf_sparsegpt.2of4


# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-sparse_2_4-loratuned

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-sparse_2_4-lora_rank128-0.5B-lr1e-4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-sparse_2_4-lora_rank128-1B-lr1e-4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-sparse_2_4-lora_rank16-0.5B-lr1e-4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-wanda_2_4-r16-alpaca-lr1e-3

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf-wanda_2_4-lora_rank16-0.5B-lr1e-3

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf_sparsegpt.2of4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/mistral-7B-v0.1-hf_wanda.2of4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/mistral-7B-v0.1-hf_sparsegpt.2of4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/mistral-7B-v0.1-hf_wanda.2of4_lora-rank16_0.5B_lr1e-3

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf_wanda.2of4

# CUDA_VISIBLE_DEVICES=${devices} python eval.py \
#     --model checkpoints/Llama-2-7b-hf_sparsegpt.2of4
