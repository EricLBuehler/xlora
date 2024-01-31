import json
import os
import pathlib
from typing import Dict, List, Optional, Union

import numpy
import peft
import safetensors  # type: ignore
import torch
import tqdm  # type: ignore
from peft.peft_model import PeftModel
from peft.tuners import lora
from transformers import PreTrainedModel  # type: ignore

from .xlora_classifier import InhibitorFlagPayload, xLoRAClassifier
from .xlora_config import xLoRAConfig
from .xlora_insertion import (
    BaseTunerWrapper,
    PeftModelWrapper,
    xLoRAConv2dLayer,
    xLoRAEmbeddingLayer,
    xLoRALinearLayer,
)


def convert_layers_to_xlora(
    base: PeftModel,
    verbose: bool,
    top_k_lora: Optional[int],
) -> int:
    """
    Returns the number of swapped layers.
    """
    assert isinstance(base.base_model, lora.LoraModel)
    total_swapped = 0

    scaling_keys = None
    for module in base.modules():
        if isinstance(module, lora.LoraLayer):
            if not scaling_keys:
                scaling_keys = list(module.scaling.keys())  # NOTE(EricLBuehler): Python 3.7: dicts are ordered!

        if isinstance(module, lora.Linear):
            assert scaling_keys is not None
            new_layer: Union[xLoRALinearLayer, xLoRAEmbeddingLayer, xLoRAConv2dLayer] = xLoRALinearLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            assert scaling_keys is not None
            new_layer = xLoRAEmbeddingLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            assert scaling_keys is not None
            new_layer = xLoRAConv2dLayer(
                model=base,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1

    if verbose:
        print(
            f"LoRA -> xLoRA complete: Swapped {total_swapped} LoRA layers (out of {len(list(base.modules()))} modules)."
        )

    return total_swapped


def add_xlora_to_model(
    model: PreTrainedModel,
    xlora_config: xLoRAConfig,
    adapters: Dict[str, str],
    verbose: bool = False,
) -> PeftModel:
    """
    This method converts all LoRA adapters to xLoRA layers, and it is one of the intended entrypoints
    for use of xLoRA. All LoRA adapters will be frozen, and the xLoRAClassifier is initialized.

    Args:
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place.
        verbose (`bool`, defaults to `False`):
            Display tqdm, total swapping count.
        adapters (`dict`):
            Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
    Returns:
        model (`PeftModel`):
            The new model.
    """

    use_trainable_adapters = xlora_config.use_trainable_adapters
    if verbose:
        adapters_items = iter(tqdm.tqdm(adapters.items()))
    else:
        adapters_items = iter(adapters.items())
    first_item = next(adapters_items)
    model_peft = PeftModel.from_pretrained(model, first_item[1], first_item[0], is_trainable=use_trainable_adapters)

    for adapter_name, model_id in adapters_items:
        model_peft.load_adapter(model_id, adapter_name, is_trainable=use_trainable_adapters)

    model_peft.base_model.set_adapter(list(adapters.keys()))

    def hook(module, *args, **kwargs) -> None:
        args_real = args[0]
        kwargs_real: dict = args[1]
        kwargs_real.update(kwargs)

        xlora_classifier: xLoRAClassifier = model_peft.internal_xlora_classifier  # type: ignore

        if "_xlora_classifier_inhibitor_flag" in kwargs_real:
            payload: InhibitorFlagPayload = kwargs_real["_xlora_classifier_inhibitor_flag"]

            del kwargs_real["_xlora_classifier_inhibitor_flag"]

            model_peft.internal_xlora_scalings = torch.full(  # type: ignore
                (payload.batch_size, payload.seq_len, xlora_classifier.n_layers, xlora_classifier.n_classes),
                payload.override_scaling_pass_value,  # requires_grad=True
            )  # TODO(EricLBuehler): is the requires_grad=True necessary?

            return

        xlora_scalings = xlora_classifier.forward(
            *args_real,
            **kwargs_real,
        )
        # Set the scalings
        model_peft.internal_xlora_scalings = xlora_scalings

    model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

    if not use_trainable_adapters:
        model_peft.base_model.eval()
        for name, param in model_peft.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

    assert isinstance(model_peft.base_model, peft.tuners.lora.LoraModel)

    total_swapped = convert_layers_to_xlora(
        model_peft,
        verbose,
        xlora_config.top_k_lora,
    )

    n_classes = len(adapters)
    xlora_classifier = xLoRAClassifier(model_peft, xlora_config, n_classes, total_swapped)

    # Setup the internal state
    base_model_wrapper = BaseTunerWrapper(model_peft.base_model, xlora_classifier)
    model_peft.base_model.forward = base_model_wrapper.forward  # type: ignore[method-assign]

    peft_model_wrapper = PeftModelWrapper(
        model_peft, model_peft.save_pretrained, xlora_config, model_peft.get_nb_trainable_parameters
    )
    model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore[method-assign]

    assert not hasattr(model_peft, "set_use_trainable_adapters")
    model_peft.set_use_trainable_adapters = peft_model_wrapper.set_use_trainable_adapters  # type: ignore

    assert not hasattr(model_peft, "print_scalings_predictions")
    model_peft.print_scalings_predictions = peft_model_wrapper.print_scalings_predictions  # type: ignore

    assert not hasattr(model_peft, "enable_scalings_logging")
    model_peft.enable_scalings_logging = peft_model_wrapper.enable_scalings_logging  # type: ignore

    assert not hasattr(model_peft, "disable_scalings_logging")
    model_peft.disable_scalings_logging = peft_model_wrapper.disable_scalings_logging  # type: ignore

    assert not hasattr(model_peft, "flush_log_scalings")
    model_peft.flush_log_scalings = peft_model_wrapper.flush_log_scalings  # type: ignore

    assert not hasattr(model_peft, "set_scaling_pass_value")
    model_peft.set_scaling_pass_value = peft_model_wrapper.set_scaling_pass_value  # type: ignore

    model_peft.get_nb_trainable_parameters = peft_model_wrapper.get_nb_trainable_parameters  # type: ignore

    model_peft.print_trainable_parameters = peft_model_wrapper.print_trainable_parameters  # type: ignore

    # Setup the model internal state
    assert not hasattr(model_peft, "internal_xlora_classifier")
    model_peft.internal_xlora_classifier = xlora_classifier

    assert not hasattr(model_peft, "internal_xlora_scalings")
    model_peft.internal_xlora_scalings = None  # type: ignore

    return model_peft


def from_pretrained(
    load_directory: str,
    model: PreTrainedModel,
    adapters: Union[List[str], Dict[str, str]],
    device: str,
    verbose: bool = False,
    from_safetensors: bool = True,
) -> PeftModel:
    """
    Loads a pretrained classifier and potentially adapters from the specified folder while initializing the model. This is the counterpart to `save_pretrained`.
    If trainable adapters was enabled, those saved adapters will be loaded.

    This method is very similar to `add_xlora_to_model`: it converts all LoRA adapters to xLoRA layers, and it is one of
    the intended entrypoints for use of xLoRA. All LoRA adapters will be frozen, and the xLoRAClassifier is initialized.

    Args:
        load_directory (`str`):
            The directory to load the classifier weights from.
        model (`PreTrainedModel`):
            The model to add the LoRA adapters to. It may be modified in place.
        adapters (`list` or `dict`):
            List of adapter names (the keys of the adapters `dict` in `add_xlora_to_model`) OR Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
            Specify the list if the adapters were trainable.
        verbose (`bool`, defaults to `False`):
            Display tqdm, total swapping count.
        device (`str`):
            Device of the model, used to load the classifier.
        from_safetensors (`bool`, *optional*, defaults to True):
            Whether to load the classifier weights from a .pt or .safetensors file.
    Returns:
        model (`PeftModel`):
            The new model.
    """

    with open(os.path.join(load_directory, "xlora_config.json"), "r") as f:
        conf = json.load(f)
        conf["device"] = torch.device(device)

        use_trainable_adapters = conf["use_trainable_adapters"]

        xlora_config = xLoRAConfig(**conf)

    if use_trainable_adapters:
        adapters_dict: Dict[str, str] = {name: os.path.join(load_directory, "adapters", name) for name in adapters}
    else:
        assert isinstance(adapters, dict)
        adapters_dict = adapters

    model_peft = add_xlora_to_model(model, xlora_config, adapters_dict, verbose)
    classifier: xLoRAClassifier = model_peft.internal_xlora_classifier  # type: ignore
    if from_safetensors:
        state_dict = safetensors.torch.load_file(  # type: ignore
            os.path.join(load_directory, "xlora_classifier.safetensors"),
            device=device,  # type: ignore
        )
    else:
        state_dict = torch.load(os.path.join(load_directory, "xlora_classifier.pt"))
    classifier.load_state_dict(state_dict)

    return model_peft


def load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]:
    """
    Load the scalings log, with awareness to the two types.

    Args:
        path (`str`):
            The path provided to `flush_log_scalings`.
        verbose (`bool`, defaults to `False`)
            Display tqdm.
    """

    output: List[torch.Tensor] = []
    if pathlib.Path(f"{path}-mapping.json").exists():
        with open(f"{path}-mapping.json", "r") as f:
            mapping: Dict[str, List[int]] = json.loads(f.read())

        mapping_full: Dict[int, torch.Tensor] = {}
        maxindex = -1

        if verbose:
            iterator = tqdm.tqdm(mapping.items())
        else:
            iterator = mapping.items()

        for file, indices in iterator:
            npy_arr = numpy.load(file)
            torch_arr = torch.from_numpy(npy_arr)
            tensors = torch_arr.split(1, dim=0)
            for tensor, index in zip(tensors, indices):
                mapping_full[index] = tensor
                if index > maxindex:
                    maxindex = index

        for index in range(maxindex + 1):
            output.append(mapping_full[index])

    else:
        npy_arr = numpy.load(path)
        torch_arr = torch.from_numpy(npy_arr)
        output.extend(torch_arr.split(1, dim=0))

    return output
