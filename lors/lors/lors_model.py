import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import LlamaForCausalLM
from lors.base.base_model import LorsBaseModel
from .lors_linear import LorsLinear
from .config import LorsConfig


class LorsModel(LorsBaseModel):
    
    def __init__(self, config, base_model):
        super().__init__(config, base_model)
        self.add_adapters()
        self.frozen_params()
        
        self.lors_linears = [m for m in self.base_model.modules() if isinstance(m, LorsLinear)]
    
    def add_adapters(self):
        linears: List[Tuple[str, nn.Linear]] = \
            [(name, module) for name, module in self.base_model.named_modules() 
                    if isinstance(module, nn.Linear) and \
                        any(name.endswith(x) for x in self.config.target_modules)]
        for name, module in linears:
            parent = self.base_model.get_submodule(".".join(name.split(".")[:-1]))
            name = name.split(".")[-1]
            new_module = LorsLinear(
                module.in_features,
                module.out_features,
                True if module.bias is not None else False,
                module.weight.device,
                module.weight.dtype,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
            )
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(parent, name, new_module)
    
    def frozen_params(self):
        for name, param in self.base_model.named_parameters():
            if not any(x in name for x in ["lora_A"]):
                param.requires_grad = False
    
    def merge_and_unload(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, LorsLinear):
                module.merge_adapter()
    
    def update_adapters(self):
        for linear in self.lors_linears:
            linear.update_adapters()

def get_lors_model(
    model: LlamaForCausalLM,
    config: LorsConfig,
):
    return LorsBaseModel(config, model)

def lors_merge_adapters(model: LorsModel):
    return model.merge_and_unload()
