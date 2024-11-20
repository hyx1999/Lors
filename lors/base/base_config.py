from dataclasses import dataclass, field
from typing import Literal, Any, List

@dataclass
class LorsBaseConfig:
    r: int = field(default=16)
    lora_alpha: float = field(default=16.0)
    target_modules: List[str] = field(
        default=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'up_proj', 'gate_proj', 'down_proj'
        ]
    )
