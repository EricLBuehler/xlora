from typing import Dict, List, Optional, Tuple

import torch
from peft.mixed_model import PeftMixedModel
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig
from transformers import PreTrainedModel

from mole.mole import mole_state
from mole.mole.mole_classifier import MoLEClassifier
from mole.mole.mole_config import MoLEConfig
from mole.mole.mole_insertion_layers import MoLELayer


def convert_layers_to_mole(
    base: PeftMixedModel,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
    top_k_lora: Optional[int] = None,
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
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward


def add_mole_to_model(
    model: PreTrainedModel,
    mole_config: MoLEConfig,
    adapters: Dict[str, str],
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
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to.
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
        if "_mole_classifier_inhibitor" in kwargs and kwargs["_mole_classifier_inhibitor"][0]:
            batch_size = kwargs["_mole_classifier_inhibitor"][1]
            mole_state.set_scalings(torch.zeros(batch_size, mole_classifier.n_classes))
            return

        mole_scalings = mole_classifier.forward(
            *args,
            **kwargs,
        )
        mole_state.set_scalings(mole_scalings)

    model.register_forward_pre_hook(hook)


def get_nb_trainable_parameters(model: PeftMixedModel) -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    model_trainable_params, model_all_param = model.get_nb_trainable_parameters()

    mole_classifier = mole_state.get_mole_classifier()
    mole_trainable_params, mole_all_param = mole_classifier.get_nb_trainable_parameters()

    trainable_params, all_param = (model_trainable_params + mole_trainable_params), (model_all_param + mole_all_param)

    return trainable_params, all_param


def print_trainable_parameters(model: PeftMixedModel):
    """
    Prints the number of trainable parameters in the model, including of the MoLE classifier.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int):
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses. Following
    this, the value of the scalings will be reset to the previous value. If the original value had a lifetime,
    only the value which it would have if it were read at assignment-time will be preserved.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    mole_state.set_scalings_lifetime(value, n_accesses_lifetime)
