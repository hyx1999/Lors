from dataclasses import dataclass, field
from peft import LoraConfig
from typing import Literal, Any
from peft import LoraConfig
from spft import SpLoraConfig, SppConfig

config_dict = {
    "lora": LoraConfig,
    "splora": SpLoraConfig,
    "splora-gc": SpLoraConfig,
    "splora-no": SpLoraConfig,
    "spp": SppConfig,
    "spp-no": SppConfig,
}


@dataclass
class AnyConfig:
    
    method: Literal[
        "lora", 
        "splora", "splora-gc", "splora-no", 
        "spp", "spp-no"
        "none"
    ] = field(default="lora")

    config: Any = field(default=None)

    def __init__(self, method: str, **kwargs):
        self.method = method
        self.config = config_dict[method]()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
