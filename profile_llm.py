import argparse
import os 
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune import check_sparsity
from utils.eval import eval_ppl, eval_zero_shot
from utils.profile_utils import profile_llm
from utils.data import get_loaders
from loss_llama import LossLlamaForCausalLM

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, args):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = 2048
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default="column")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    device = torch.device("cuda:0")
    print("use device ", device)
    
    sparsity = check_sparsity(model)
    print(f"sparsity = {sparsity}")
    # model.to(device)
    
    # # Get the test loader
    _, testloader = get_loaders("slimpajama", seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    selected_indices = profile_llm(model, testloader, device, args)
    
    assert args.output_path is not None
    torch.save(selected_indices, args.output_path)


if __name__ == '__main__':
    main()
