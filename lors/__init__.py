from peft import LoraConfig, get_peft_model
from .base import LorsBaseConfig, LorsBaseModel
from .lors import (
    LorsConfig,
    get_lors_model,
)
from .splora import (
    SpLoraConfig, 
    get_splora_model,
    splora_methods, 
)
from .spp import (
    SppConfig, 
    get_spp_model,
    spp_methods,
)
from .dispatch_config import DispatchConfig

def get_lors_model(model, config: DispatchConfig):
    if config.method in splora_methods.keys():
        return get_splora_model(model, config.config)
    elif config.method in spp_methods.keys():
        return get_spp_model(model, config.config)
    elif config.method == "lors":
        return get_lors_model(model, config.config)
    else:
        raise ValueError

def get_model_with_adapters(model, config: DispatchConfig, adapter_name: str = "default", mixed: bool = False):
    if config.method == "none":
        return model
    elif config.method == "lora":
        return get_peft_model(model, config.config, adapter_name, mixed)
    else:
        raise get_lors_model(model, config)

def merge_and_unload(model, config: DispatchConfig):
    if config.config.method == "none":
        return model
    elif config.config.method == "lora":
        return model.base_model.merge_and_unload()
    else:
        return model.merge_and_unload()
