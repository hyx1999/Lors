import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from .modules import ExtLoraConfig, splora_modules, spp_modules
from collections import defaultdict
from typing import Optional, List, Tuple
from tqdm import tqdm
from accelerate import PartialState

def get_splora_model(
    model: LlamaForCausalLM, 
    config: ExtLoraConfig, 
):
    setattr(model, "peft_config", config)
    linears: List[Tuple[str, nn.Linear]] = \
        [(name, module) for name, module in model.named_modules() 
                if isinstance(module, nn.Linear) and any(name.endswith(x) for x in config.target_modules)]
    LinearModule = splora_modules[config.method]
    for name, module in tqdm(linears, disable=not PartialState.is_main_process):
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        name = name.split(".")[-1]
        new_module = LinearModule(
            module.in_features,
            module.out_features,
            True if module.bias is not None else False,
            module.weight.device,
            module.weight.dtype,
            r=config.r,
            lora_alpha=config.lora_alpha,
        )
        new_module.load_state_dict(module.state_dict(), strict=False)
        new_module.set_mask()
        setattr(parent, name, new_module)
    for name, param in model.named_parameters():
        if not any(x in name for x in ["lora_A", "lora_B"]):
            param.requires_grad = False
    return model

def splora_merge_adapters(model: LlamaForCausalLM):
    for name, module in model.named_modules():
        if isinstance(module, tuple(splora_modules.values())):
            module.merge_adapter()

def get_spp_model(
    model: LlamaForCausalLM, 
    config: ExtLoraConfig, 
):
    setattr(model, "peft_config", config)
    linears: List[Tuple[str, nn.Linear]] = \
        [(name, module) for name, module in model.named_modules() 
                if isinstance(module, nn.Linear) and any(name.endswith(x) for x in config.target_modules)]
    LinearModule = spp_modules[config.method]
    for name, module in tqdm(linears, disable=not PartialState.is_main_process):
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        name = name.split(".")[-1]
        new_module = LinearModule(
            module.in_features,
            module.out_features,
            r=config.r,
            lora_alpha=config.lora_alpha,
            bias=True if module.bias is not None else False,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_module.load_state_dict(module.state_dict(), strict=False)
        setattr(parent, name, new_module)
    for name, param in model.named_parameters():
        if not any(x in name for x in ["lora_A", "lora_B"]):
            param.requires_grad = False
    return model

def spp_merge_adapters(model: LlamaForCausalLM):
    for name, module in model.named_modules():
        if isinstance(module, tuple(spp_modules.values())):
            module.merge_adapter()
