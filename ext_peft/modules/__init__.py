from .splora_linear import SpLoraLinear
from .splora_linear_gradckpt import SpLoraGradckptLinear
from .splora_linear_naive import SpLoraNaiveLinear
# from .splora_linear_mask import SpLoraMaskLinear as SpLoraLinear
from .spp_linear import SppLinear
from .spp_linear_naive import SppNaiveLinear
from .ext_lora_config import ExtLoraConfig

splora_modules = {
    "splora": SpLoraLinear,
    "splora-gradckpt": SpLoraGradckptLinear,
    "splora-naive": SpLoraNaiveLinear,
}

spp_modules = {
    "spp": SppLinear,
    "spp-naive": SppNaiveLinear,    
}
