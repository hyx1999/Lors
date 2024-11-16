from peft import LoraConfig, get_peft_model
from .splora import (
    SpLoraConfig, 
    splora_modules, 
    get_splora_model, 
    splora_merge_adapters
)
from .spp import (
    SppConfig, 
    spp_modules, 
    get_spp_model,
    spp_merge_adapters
)
from .any_config import AnyConfig

def get_lors_model(model, peft_config):
    if peft_config.method in splora_modules.keys():
        return get_splora_model(model, peft_config)
    elif peft_config.method in spp_modules.keys():
        return get_spp_model(model, peft_config)
    else:
        raise ValueError

def lors_merge_and_unload(model, peft_config):
    if peft_config.method in splora_modules.keys():
        return splora_merge_adapters(model)
    elif peft_config.method in spp_modules.keys():
        return spp_merge_adapters(model)
    else:
        raise ValueError


def get_peft_and_lors_model(model, any_config: AnyConfig, adapter_name: str = "default", mixed: bool = False):
    if any_config.config.method == "none":
        return model
    elif any_config.config.method == "lora":
        return get_peft_model(model, any_config.config, adapter_name, mixed)
    else:
        raise get_lors_model(model, any_config.config)


def merge_and_unload(model, any_config: AnyConfig):
    if any_config.config.method == "none":
        return model
    elif any_config.config.method == "lora":
        return model.base_model.merge_and_unload()
    else:
        return lors_merge_and_unload(model, any_config.config)
