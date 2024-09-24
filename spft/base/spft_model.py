import torch
import torch.nn as nn

class SpftModel(nn.Module):
    
    def __init__(self, config, base_model) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
