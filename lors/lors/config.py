from lors.base.base_config import LorsBaseConfig
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class LorsConfig(LorsBaseConfig):
    sparsify_update_freq = field(default=256)
    grad_update_freq = field(default=128)
