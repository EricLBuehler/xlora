import json
import os
from typing import Dict, List, Optional

import safetensors
import torch
import tqdm
from peft.mixed_model import PeftMixedModel
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig
from transformers import PreTrainedModel

from mole.mole_model import MoLEModel

from . import mole_state
from .mole_classifier import MoLEClassifier
from .mole_config import MoLEConfig
from .mole_insertion_layers import MoLELayer


def convert_layers_to_mole(
    base: PeftMixedModel,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    verbose: bool,
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
    top_k_lora: Optional[int] = None,
):
    modules = list(base.modules())
    if not verbose:
        iterable = modules
    else:
        iterable = tqdm.tqdm(modules)
    total_swapped = 0
    for module in iterable:
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
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward
            total_swapped += 1
    if verbose:
        print(f"Swapped {total_swapped} layers.")


def add_mole_to_model(
    model: PreTrainedModel,
    mole_config: MoLEConfig,
    adapters: Dict[str, str],
    verbose: bool,
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
) -> MoLEModel:
    """
    This method converts all LoRA adapters to MoLE layers, and it is one of the intended entrypoints
    for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

    When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
    the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
    errors.

    Args:
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place.
        verbose (`bool`):
            Display tqdm, total swapping count.
        adapters (`dict`):
            Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
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
    Returns:
        model (`MoLEModel`):
            The new model.
    """

    adapters_items = list(adapters.items())
    first_item = adapters_items[0]
    adapters_items = adapters_items[1:]
    model = PeftMixedModel.from_pretrained(model, first_item[1], first_item[0], False)
    for adapter_name, model_id in adapters_items:
        model.load_adapter(model_id, adapter_name)

    peft_config = model.peft_config
    adapters = list(adapters.keys())

    convert_layers_to_mole(
        model,
        adapters,
        verbose,
        peft_config,
        combination_type,
        svd_rank,
        svd_clamp,
        svd_full_matrices,
        svd_driver,
        top_k_lora=mole_config.top_k_lora,
    )

    n_classes = len(adapters)
    mole_classifier = MoLEClassifier(model, mole_config, n_classes)
    mole_state.set_mole_classifier(mole_classifier)

    def hook(module, *args, **kwargs) -> None:
        if "_mole_classifier_inhibitor_flag" in kwargs:
            assert isinstance(kwargs["_mole_classifier_inhibitor_flag"], int)
            batch_size = kwargs["_mole_classifier_inhibitor_flag"]
            mole_state.set_scalings(torch.zeros(batch_size, mole_classifier.n_classes))
            return

        mole_scalings = mole_classifier.forward(
            *args,
            **kwargs,
        )
        mole_state.set_scalings(mole_scalings)

    model.register_forward_pre_hook(hook)

    for param in model.base_model.parameters():
        param.requires_grad = False

    return MoLEModel(model)


def from_pretrained(
    load_directory: str,
    from_safetensors: bool,
    model: PreTrainedModel,
    mole_config: MoLEConfig,
    verbose: bool,
    adapters: Dict[str, str],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
) -> MoLEModel:
    """
    Loads a pretrained classifier from the specified folder while initializing the model. This is the counterpart to `MoLEModel.save_pretrained`.

    This method is very similar to `add_mole_to_model`: it converts all LoRA adapters to MoLE layers, and it is one of
    the intended entrypoints for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

    When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
    the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
    errors.

    Args:
        load_directory (`str`):
            The directory to load the classifier weights from.
        from_safetensors (`bool`):
            Whether to load the classifier weights from a .pt or .safetensors file.
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place.
        verbose (`bool`):
            Display tqdm, total swapping count.
        adapters (`dict`):
            Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
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
    Returns:
        model (`MoLEModel`):
            The new model.
    """

    model = add_mole_to_model(
        model, mole_config, adapters, verbose, combination_type, svd_rank, svd_clamp, svd_full_matrices, svd_driver
    )

    classifier = mole_state.get_mole_classifier()
    with open(os.path.join(load_directory, "mole_classifier_config.json"), "w") as f:
        conf = json.load(f)
        assert classifier.n_classes == conf["n_classes"]

    if from_safetensors:
        state_dict = safetensors.torch.load_file(
            os.path.join(load_directory, "mole_classifier.safetensors"),
            device={k: v.device for k, v in classifier.state_dict()},
        )
    else:
        state_dict = torch.load(os.path.join(load_directory, "mole_classifier.pt"))
    classifier.load_state_dict(state_dict)

    return model


def set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int):
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses. Following
    this, the value of the scalings will be reset to the previous value. If the original value had a lifetime,
    only the value which it would have if it were read at assignment-time will be preserved.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    mole_state.set_scalings_lifetime(value, n_accesses_lifetime)
