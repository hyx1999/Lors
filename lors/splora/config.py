from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class SpLoraConfig:
    r: int = 16
    lora_alpha: float = 16.0
