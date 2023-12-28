from typing import Dict, List, Optional
import torch.nn as nn

from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig

from mole.mole_insertion_layers import MoLELayer


def convert_layers_to_mole(
    base: nn.Module,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
):
    modules = list(base.modules())
    for module in modules:
        if isinstance(module, lora.LoraLayer):
            new_layer = MoLELayer(
                adapters=adapters,
                target=module,
                peft_config=peft_config,
                combination_type=combination_type,
                svd_rank=svd_rank,
                svd_clamp=svd_clamp,
                svd_full_matrices=svd_full_matrices,
                svd_driver=svd_driver,
            )
            module.forward = new_layer.forward
