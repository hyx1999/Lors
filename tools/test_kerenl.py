import torch
from ext_peft.modules import SppLinear

fc = SppLinear(32, 16, bias=False)
with torch.no_grad():
    fc.lora_A.weight.add_(torch.rand_like(fc.lora_A.weight))
    fc.lora_A.weight_col.add_(torch.rand_like(fc.lora_A.weight_col))

    x = torch.randn(1, 32)
    y1 = fc(x)
    
    print(y1)
    
    fc.merge_adapter()
    
    y2 = fc(x)
    print(y2)

    print((y1 - y2).norm())
    

