import torch
import torch.nn as nn
import os
import logging
from tqdm import tqdm
from utils.prune import find_layers
from typing import Dict


def profile_llm(model, testenc, dev, args):

    model.to(dev)
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.model.gradient_checkpointing = True
    
    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = 1
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    layers: Dict[str, nn.Linear] = find_layers(model)
    metrics: Dict[str, torch.Tensor] = {k: None for k in layers.keys()}

    for i in tqdm(range(nbatches)):
        outputs = model(input_ids=input_ids[i], labels=input_ids[i])
        loss = outputs.loss
        loss.backward()
        with torch.no_grad():
            for name, module in layers.items():
                grad = module.weight.grad
                mask = (module.weight != 0)
                if metrics[name] is None:
                    metrics[name] = grad.abs() * mask
                else:
                    metrics[name] += grad.abs() * mask
                del module.weight.grad

    model.config.use_cache = use_cache
    
    selected_indices = {}
    
    if args.mode == "block":
        block_size = args.block_size
        for name, metric in metrics.items():
            n, m = metric.shape
            metric = metric\
                .reshape(n // block_size, block_size, m // block_size, block_size)\
                .permute(0, 2, 1, 3)\
                .reshape((n // block_size) * (m // block_size), block_size * block_size)\
                .mean(dim=-1).cpu()
            indices = torch.sort(metric, stable=True, descending=True)[1]
            indices = torch.tensor([[i // (m // block_size), i % (m // block_size)] for i in indices])
            selected_indices[name] = indices
    elif args.mode == "column":
        block_size = args.block_size
        for name, metric in metrics.items():
            n, m = metric.shape
            metric = metric.mean(dim=0).cpu()
            indices = torch.sort(metric, stable=True, descending=True)[1]
            selected_indices[name] = indices
    else:
        raise ValueError

    return selected_indices
