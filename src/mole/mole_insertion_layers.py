from ast import List
from typing import Any
from peft.tuners import lora
from torch import Tensor

from mole import mole_state


class MoLELinear:
    def __init__(self, adapters: List[lora.Linear]) -> None:
        # TODO(EricLBuehler): Freeze the LoRA adapters
        for adapter in adapters:
            pass
        self.adapters = adapters

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        # TODO(EricLBuehler): Combine the LoRA adapters using the scaling
        _scaling = mole_state.get_scalings()
        pass
