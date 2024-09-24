#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-7b-hf \
#     --sparsity_type 2:4 \
#     --prune_method wanda \
#     --prune_groups qkv,o,ug,d \
#     --save log/wanda \
#     --save_model checkpoints/tmp

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-7b-hf \
#     --sparsity_type 2:4 \
#     --prune_method sparsegpt \
#     --prune_groups qkv,o,ug,d \
#     --save log/sparsegpt \
#     --save_model checkpoints/Llama-2-7b-hf-sparsegpt_2_4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-13b-hf \
#     --sparsity_type 2:4 \
#     --prune_method wanda \
#     --prune_groups qkv,o,ug,d \
#     --save log/wanda \
#     --save_model checkpoints/Llama-2-13b-hf-wanda_2_4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-13b-hf \
#     --sparsity_type 2:4 \
#     --prune_method sparsegpt \
#     --prune_groups qkv,o,ug,d \
#     --save log/sparsegpt \
#     --save_model checkpoints/Llama-2-13b-hf-sparsegpt_2_4

CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
    --model checkpoints/qwen2-7b \
    --sparsity_type 2:4 \
    --prune_method wanda \
    --save log/wanda \
    --save_model checkpoints/qwen2-7b_wanda.2of4
