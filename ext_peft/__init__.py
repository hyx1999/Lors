from peft import LoraConfig, get_peft_model
from .modules import spp_modules, splora_modules, ExtLoraConfig
from .utils import (
    get_splora_model, 
    splora_merge_adapters, 
    get_spp_model, 
    spp_merge_adapters
)

def ext_get_peft_model(model, peft_config, adapter_name: str = "default", mixed: bool = False):
    if peft_config.method == "lora":
        return get_peft_model(model, peft_config, adapter_name, mixed)
    elif peft_config.method in splora_modules.keys():
        return get_splora_model(model, peft_config)
    elif peft_config.method in spp_modules.keys():
        return get_spp_model(model, peft_config)
    elif peft_config.method == "none":
        return model
    else:
        raise ValueError


def ext_merge_and_unload(model, peft_config):
    if peft_config.method == "lora":
        return model.base_model.merge_and_unload()
    elif peft_config.method in splora_modules.keys():
        splora_merge_adapters(model)
        return model
    elif peft_config.method in spp_modules.keys():
        spp_merge_adapters(model)
        return model
    elif peft_config.method == "none":
        return model
    else:
        raise ValueError
