import torch.nn as nn

from peft.tuners import lora

from mole.mole_insertion_layers import MoLELinear


def convert_layers_to_mole(base: nn.Module):
    modules = list(base.modules())
    for module in modules:
        if isinstance(module, lora.Linear):
            new_layer = MoLELinear(
                [module]
            )  # TODO(EricLBuehler): Use the passed LoRA adapters for this layer
            module.forward = new_layer.forward
