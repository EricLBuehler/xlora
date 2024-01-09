import collections
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import peft
import safetensors  # type: ignore
import torch
import tqdm  # type: ignore
from peft.peft_model import PeftModel
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig  # type: ignore
from transformers import PreTrainedModel  # type: ignore

from . import mole_state
from .mole_classifier import MoLEClassifier
from .mole_config import MoLEConfig
from .mole_insertion import BaseTunerWrapper, MoLELayer


def convert_layers_to_mole(
    base: PeftModel,
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
                target_forward=module.forward,
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
) -> PeftModel:
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
        model (`PeftModel`):
            The new model.
    """

    def hook(module, *args, **kwargs) -> None:
        args_real = args[0]
        kwargs_real: dict = args[1]
        kwargs_real.update(kwargs)

        if "_mole_classifier_inhibitor_flag" in kwargs_real:
            assert isinstance(kwargs_real["_mole_classifier_inhibitor_flag"], int)
            batch_size = kwargs_real["_mole_classifier_inhibitor_flag"]
            del kwargs_real["_mole_classifier_inhibitor_flag"]
            mole_state.set_scalings(torch.zeros(batch_size, mole_classifier.n_classes))
            return

        mole_scalings = mole_classifier.forward(
            *args_real,
            **kwargs_real,
        )
        mole_state.set_scalings(mole_scalings)

    def after_hook(module, input, output):
        print(f"{output=}")

    model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)
    model.register_forward_hook(after_hook)

    adapters_items = list(adapters.items())
    first_item = adapters_items[0]
    adapters_items = adapters_items[1:]
    model_peft = PeftModel.from_pretrained(model, first_item[1], first_item[0], False)
    for adapter_name, model_id in adapters_items:
        model.load_adapter(model_id, adapter_name)

    assert isinstance(model_peft.base_model, peft.tuners.lora.LoraModel)

    base_model_wrapper = BaseTunerWrapper(model_peft.base_model)
    model_peft.base_model.forward = base_model_wrapper.forward  # type: ignore[method-assign]

    peft_config = model_peft.peft_config
    adapters_keys: List[str] = list(adapters.keys())

    convert_layers_to_mole(
        model_peft,
        adapters_keys,
        peft_config,
        verbose,
        combination_type,
        svd_rank,
        svd_clamp,
        svd_full_matrices,
        svd_driver,
        top_k_lora=mole_config.top_k_lora,
    )

    n_classes = len(adapters)
    mole_classifier = MoLEClassifier(model_peft, mole_config, n_classes)
    mole_state.set_mole_classifier(mole_classifier)

    for param in model.base_model.parameters():
        param.requires_grad_(False)

    return model_peft


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
) -> PeftModel:
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
        model (`PeftModel`):
            The new model.
    """

    model_peft = add_mole_to_model(
        model, mole_config, adapters, verbose, combination_type, svd_rank, svd_clamp, svd_full_matrices, svd_driver
    )

    classifier = mole_state.get_mole_classifier()
    with open(os.path.join(load_directory, "mole_classifier_config.json"), "w") as f:
        conf = json.load(f)
        assert classifier.n_classes == conf["n_classes"]

    if from_safetensors:
        state_dict = safetensors.torch.load_file(  # type: ignore
            os.path.join(load_directory, "mole_classifier.safetensors"),
            device={k: v.device for k, v in classifier.state_dict()},  # type: ignore
        )
    else:
        state_dict = torch.load(os.path.join(load_directory, "mole_classifier.pt"))
    classifier.load_state_dict(state_dict)

    return model_peft


def set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int):
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses. Following
    this, the value of the scalings will be reset to the previous value. If the original value had a lifetime,
    only the value which it would have if it were read at assignment-time will be preserved.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    mole_state.set_scalings_lifetime(value, n_accesses_lifetime)


def save_pretrained(
    save_directory: str,
    safe_serialization: Optional[bool] = True,
    is_main_process: bool = True,
) -> None:
    r"""
    This function saves the classifier weights to a directory. It is the counerpart to `from_pretrained`.

    Args:
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        safe_serialization (`bool`, *optional*):
            Whether to save the adapter files in safetensors format, defaults to `True`.
        is_main_process (`bool`, *optional*):
            Whether the process calling this is the main process or not. Will default to `True`. Will not save the
            checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
    """
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

    classifier = mole_state.get_mole_classifier()

    conf = {"n_classes": classifier.n_classes}
    with open(os.path.join(save_directory, "mole_classifier_config.json"), "w") as f:
        json.dump(conf, f)

    state_dict = classifier.state_dict()
    if safe_serialization:
        # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L223
        if is_main_process and safe_serialization:
            # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs: collections.defaultdict[
                Union[Tuple[torch.device, int, int], int], List[str]
            ] = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[peft.utils.other.id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors.
            shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            for _, names in shared_ptrs.items():
                # Here we just clone the shared tensors to avoid tensor aliasing which is
                # not supported in safetensors.
                for shared_tensor_name in names[1:]:
                    state_dict[shared_tensor_name] = state_dict[shared_tensor_name].clone()

            safetensors.torch.save_file(  # type: ignore
                state_dict, os.path.join(save_directory, "mole_classifier.safetensors"), metadata={"format": "pt"}
            )
    elif is_main_process:
        torch.save(state_dict, os.path.join(save_directory, "mole_classifier.pt"))


def get_nb_trainable_parameters(model: PeftModel) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    model_trainable_params, model_all_param = model.get_nb_trainable_parameters()

    mole_classifier = mole_state.get_mole_classifier()
    mole_trainable_params, mole_all_param = mole_classifier.get_nb_trainable_parameters()

    trainable_params, all_param = (
        (model_trainable_params + mole_trainable_params),
        (model_all_param + mole_all_param),
    )

    return trainable_params, all_param


def print_trainable_parameters(model: PeftModel):
    """
    Prints the number of trainable parameters in the model, including of the MoLE classifier.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
