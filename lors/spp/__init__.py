import torch
import torch.nn as nn
from typing import List, Tuple

from .spp_linear import SppLinear
from .spp_linear_no import SppLinearNO
from .config import SppConfig
from lors.base.lors_model import LorsModel

spp_modules = {
    "spp": SppLinear,
    "spp-naive": SppLinearNO,    
}

def get_spp_model(
    model, 
    config, 
):
    linears: List[Tuple[str, nn.Linear]] = \
        [(name, module) for name, module in model.named_modules() 
                if isinstance(module, nn.Linear) and any(name.endswith(x) for x in config.target_modules)]
    LinearModule = spp_modules[config.method]
    for name, module in linears:
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
    return LorsModel(config, model)

def spp_merge_adapters(model):
    for name, module in model.named_modules():
        if isinstance(module, tuple(spp_modules.values())):
            module.merge_adapter()
    return model.base_model
