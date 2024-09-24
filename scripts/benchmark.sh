#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

CUDA_VISIBLE_DEVICES=${devices} python benchmark.py \
    --base_model checkpoints/Llama-2-7b-hf \
    --sparse_model checkpoints/Llama-2-7b-hf-sparse_2_4-r512-fuse-tuned \
    --cutlass \
    --batch_size 8 \
    --length 256
