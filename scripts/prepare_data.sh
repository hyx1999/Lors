#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

# python prepare_data.py \
#     --num_samples 500000 \
#     --output_path /data/SlimPajama-0.5B-ap

python prepare_data.py \
    --num_samples 1000000 \
    --output_path /data/SlimPajama-1B-ap
