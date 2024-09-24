import argparse
import os 
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune import prune_model
from utils.eval import eval_ppl, eval_zero_shot
from utils.eval_utils import evaluator
from utils.data import get_loaders
from torch.sparse import to_sparse_semi_structured

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
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
    parser.add_argument("--sparsity_type", type=str, default="2:4", choices=["2:4", "unstructured"])
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--dump_sparse_model', action="store_true")
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    else:
        prune_n = prune_m = None

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    print("use device ", device)
    
    prune_model(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    
    # ppl_test = eval_ppl(args, model, tokenizer, device)
    # print(f"wikitext perplexity {ppl_test}")

    # Get the test loader
    _, testloader = get_loaders("wikitext2", seed=0, seqlen=model.seqlen, tokenizer=tokenizer)
    ppl_test = evaluator(model, testloader, device, args)
    print(f"wikitext perplexity: {ppl_test}")
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    format_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_filepath = os.path.join(args.save, f"log_{model_name}_{format_time}.txt")
    with open(save_filepath, "w") as f:
        print(f"ppl_test: {ppl_test:.4f}\n\n", file=f, flush=True)
        print(f"args:", file=f, flush=True)
        for k, v in vars(args).items():
            print(f"\t{k}={v}", file=f, flush=True)
    
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

if __name__ == '__main__':
    main()
