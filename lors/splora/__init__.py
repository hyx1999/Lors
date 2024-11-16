import torch
import torch.nn as nn
from typing import List, Tuple

from .splora_linear import SpLoraLinear
from .splora_linear_gc import SpLoraLinearGC
from .splora_linear_no import SpLoraLinearNO
from .config import SpLoraConfig
from lors.base.lors_model import LorsModel

splora_modules = {
    "splora": SpLoraLinear,
    "splora-gc": SpLoraLinearGC,
    "splora-no": SpLoraLinearNO,
}

def get_splora_model(
    model, 
    config, 
):
    linears: List[Tuple[str, nn.Linear]] = \
        [(name, module) for name, module in model.named_modules() 
                if isinstance(module, nn.Linear) and any(name.endswith(x) for x in config.target_modules)]
    LinearModule = splora_modules[config.method]
    for name, module in linears:
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
    return LorsModel(config, model)

def splora_merge_adapters(model):
    for name, module in model.named_modules():
        if isinstance(module, tuple(splora_modules.values())):
            module.merge_adapter()
    return model.base_model
