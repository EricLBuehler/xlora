from typing import Any
from peft.tuners import lora
from torch import Tensor

class MoLELinear(lora.Linear):
    def __init__(self, other: lora.Linear) -> None:
        for attr in dir(other):
            if not hasattr(self, attr):
                setattr(self, attr, getattr(other, attr))

    def forward(self, x: Tensor, scaling: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                # scaling = self.scaling[active_adapter] # This is the only change
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result