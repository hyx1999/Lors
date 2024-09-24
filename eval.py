import argparse
import os 
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune import check_sparsity
from utils.eval import eval_ppl, eval_zero_shot
from utils.eval_utils import evaluator
from utils.eval_humaneval import eval_humaneval
from utils.data import get_loaders

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, 
        default="nlu", 
        choices=[
            "nlu", 
            "gsm8k",
            "mt-bench",
            "human-eval",
        ]
    )
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--model_name', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda:0")
    print("use device ", device)
    
    # sparsity = check_sparsity(model)
    # print(f"sparsity = {sparsity}")
    # model.to(device)
    
    # # Get the test loader
    if args.eval_ppl:
        _, testloader = get_loaders("wikitext2", seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
        ppl_test = evaluator(model, testloader, device, args)
        print(f"wikitext perplexity: {ppl_test}")

    model.to(device)
    
    if args.task == "nlu":   
        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, False)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
    elif args.task == "gsm8k":
        task_list = [args.task]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, False)
        print("********************************")
        print("gsm8k results")
        print(results)
    elif args.task == "human-eval":
        eval_humaneval(args, model, tokenizer)
    else:
        raise NotImplementedError
    

if __name__ == '__main__':
    main()
