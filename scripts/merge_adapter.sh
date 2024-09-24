#!/bin/bash
set -e
set -x

cd $(dirname $0)/..


python merge_adapter.py \
    --model checkpoints/Llama-2-7b-hf-sparse_2_4-wanda-r512 \
    --adapter checkpoints/Llama-2-7b-hf-sparse_2_4-wanda-r512-adapter \
    --output_dir checkpoints/Llama-2-7b-hf-sparse_2_4-wanda-r512-finetuned
