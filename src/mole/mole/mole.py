from typing import Dict, List, Optional

import torch.nn as nn
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig

from mole.mole import mole_state
from mole.mole.mole_classifier import MoLEClassifier, MoLEClassifierConfig
from mole.mole.mole_insertion_layers import MoLELayer


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


def add_mole_to_model(
    model: nn.Module,
    mole_config: MoLEClassifierConfig,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
):
    """
    This method converts all LoRA adapters to MoLE layers, and it is the intended entrypoint
    for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

    When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
    the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
    errors.

    Args:
        base (`Module`):
            The model to recursively loop over to find and convert all LoRA adapters.
        adapters (`list`):
            List of adapter names to be merged.
        peft_config: (`dict`):
            PeftConfigs for each adapter in the LoraLayer.
        combination_type (`str`):
            Type of merging. Can be one of [`svd`, `linear`, `cat`]. When using the `cat` combination_type you
            should be aware that rank of the resulting adapter will be equal to the sum of all adapters ranks. So
            it's possible that the mixed adapter may become too big and result in OOM errors.
        svd_rank (`int`, *optional*):
            Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
        svd_clamp (`float`, *optional*):
            A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
            clamping. Defaults to None.
        svd_full_matrices (`bool`, *optional*):
            Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
            tensors U and Vh. Defaults to True.
        svd_driver (`str`, *optional*):
            Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
            one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
            documentation. Defaults to None.
    """
    convert_layers_to_mole(
        model, adapters, peft_config, combination_type, svd_rank, svd_clamp, svd_full_matrices, svd_driver
    )

    mole_classifier = MoLEClassifier(mole_config)

    def hook(module, *args, **kwargs) -> None:
        mole_output = mole_classifier.forward(
            *args,
            **kwargs,
        )
        mole_scalings = list(mole_output)
        mole_state.set_scalings(mole_scalings)

    model.register_forward_pre_hook(hook)
