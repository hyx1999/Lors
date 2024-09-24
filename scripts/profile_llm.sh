#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=3

CUDA_VISIBLE_DEVICES=${devices} python profile_llm.py \
    --model checkpoints/Llama-2-7b-hf-sparse_2_4 \
    --output_path checkpoints/Llama-2-7b-hf-sparse_2_4/selected_indices_slimpajama.pth
