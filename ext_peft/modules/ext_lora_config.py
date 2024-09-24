from dataclasses import dataclass, field
from peft import LoraConfig
from typing import Literal

@dataclass
class ExtLoraConfig(LoraConfig):
    
    method: Literal["lora", "splora", "spp", "none"] = field(default="lora")