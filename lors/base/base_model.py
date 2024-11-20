import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from .base_config import LorsBaseConfig

class LorsBaseModel(nn.Module):
    
    def __init__(self, 
        config: LorsBaseConfig, 
        base_model: LlamaForCausalLM,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def merge_and_unload(self):
        pass
    
    def update_adapters(self):
        pass
