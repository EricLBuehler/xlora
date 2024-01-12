from typing import Any, Callable, List, Optional

import torch
from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner  # type: ignore
from torch import Tensor

from mole import mole_state


class MoLELayer:
    """
    A MoLELayer wraps any LoraLayer and performs the MoLE operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from mole_state to execute the
    MoLE algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        top_k_lora: Optional[int] = None,
    ) -> None:
        self.target_forward = target_forward
        self.target = target
        self.scaling_keys = scaling_keys
        self.top_k_lora = top_k_lora

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the MoLELayer class).
        """
        old_scalings = self.target.scaling.copy()

        outputs: List[Tensor] = []
        if self.top_k_lora is None:
            for i, (batch_x, batch_scalings) in enumerate(zip(x, mole_state.get_scalings())):
                self.scale_adapters(self.target, batch_scalings, self.scaling_keys)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)

                self.target.scaling = old_scalings
        else:
            for i, (batch_x, batch_scalings) in enumerate(zip(x, mole_state.get_scalings())):
                (topk_scalings, indices) = torch.topk(input=batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.scaling_keys[i] for i in indices]

                self.scale_adapters(self.target, topk_scalings, adapters)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)

                self.target.scaling = old_scalings

        return torch.cat(outputs, dim=0)

    @staticmethod
    def scale_adapters(target: lora.LoraLayer, scalings: Tensor, adapters: List[str]):
        for scaling, adapter in zip(scalings, adapters):
            target.scaling[adapter] = target.scaling[adapter] * scaling


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
