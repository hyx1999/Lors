import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import LlamaForCausalLM
from lors.base.base_model import LorsBaseModel
from .lors_linear import LorsLinear
from .config import LorsConfig
from .utils import SparsifyFn, GradFn, ScaleFn

class LorsModel(LorsBaseModel):
    
    def __init__(self, config, base_model):
        super().__init__(config, base_model)
        scale_fn = ScaleFn()
        sparsify_fn = SparsifyFn(config.sparsify_update_freq)
        grad_fn = GradFn(config.grad_update_freq)
        self.add_adapters(scale_fn, sparsify_fn, grad_fn)
        self.frozen_params()
        self.scale_fn = scale_fn
        self.sparsify_fn = sparsify_fn
        self.grad_fn = grad_fn
        
        self.lors_linears = [m for m in self.base_model.modules() if isinstance(m, LorsLinear)]
    
    def add_adapters(self, scale_fn: ScaleFn, sparsify_fn: SparsifyFn, grad_fn: GradFn):
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
                scale_fn=scale_fn,
                sparsify_fn=sparsify_fn,
                grad_fn=grad_fn,
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

    def update_substep(self):
        self.scale_fn.update_substep()
        self.sparsify_fn.update_substep()
        self.grad_fn.update_substep()
    
    def update_step(self):
        for linear in self.lors_linears:
            linear.update_adapters()
        self.scale_fn.update_step()
        self.sparsify_fn.update_step()
        self.grad_fn.update_step()

def get_lors_model(
    model: LlamaForCausalLM,
    config: LorsConfig,
):
    return LorsBaseModel(config, model)

def lors_merge_adapters(model: LorsModel):
    return model.merge_and_unload()
