from typing import Any, Callable, List

import torch
from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner  # type: ignore
from torch import Tensor

from mole.mole_classifier import _MOLE_ADAPTER_NAME


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
    ) -> None:
        self.target_forward = target_forward
        self.target = target

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the MoLELayer class).
        """
        outputs: List[Tensor] = []
        for i, batch_x in enumerate(x):
            self.target.set_adapter(_MOLE_ADAPTER_NAME + f"_{i}")

            output = self.target_forward(batch_x, *args, **kwargs)
            outputs.append(output)

        return torch.cat(outputs, dim=0)


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
