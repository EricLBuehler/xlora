import json
import os
from typing import Dict, Optional, Tuple

import peft
import safetensors  # type: ignore
import torch
import tqdm  # type: ignore
from peft.peft_model import PeftModel
from peft.tuners import lora
from transformers import PreTrainedModel  # type: ignore

from xlora import xlora_classifier

from . import xlora_state
from .xlora_classifier import xLoRAClassifier
from .xlora_config import xLoRAConfig
from .xlora_insertion import BaseTunerWrapper, PeftModelWrapper, xLoRALayer


def convert_layers_to_xlora(
    base: PeftModel,
    verbose: bool,
    top_k_lora: Optional[int] = None,
) -> int:
    """
    Returns the number of swapped layers.
    """
    assert isinstance(base.base_model, lora.LoraModel)
    modules = list(base.modules())
    if not verbose:
        iterable = modules
    else:
        iterable = tqdm.tqdm(modules)
    total_swapped = 0

    scaling_keys = None
    for module in iterable:
        if isinstance(module, lora.LoraLayer):
            if not scaling_keys:
                scaling_keys = list(module.scaling.keys())  # NOTE(EricLBuehler): Python 3.7: dicts are ordered!
            new_layer = xLoRALayer(
                target=module,
                target_forward=module.forward,
                scaling_keys=scaling_keys,
                top_k_lora=top_k_lora,
                layer_number=total_swapped,
            )
            module.forward = new_layer.forward
            total_swapped += 1
    if verbose:
        print(f"Swapped {total_swapped} layers.")

    return total_swapped


def add_xlora_to_model(
    model: PreTrainedModel,
    xlora_config: xLoRAConfig,
    adapters: Dict[str, str],
    verbose: bool,
) -> PeftModel:
    """
    This method converts all LoRA adapters to xLoRA layers, and it is one of the intended entrypoints
    for use of xLoRA. All LoRA adapters will be frozen, and the xLoRAClassifier is initialized.

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
    Returns:
        model (`PeftModel`):
            The new model.
    """

    def hook(module, *args, **kwargs) -> None:
        args_real = args[0]
        kwargs_real: dict = args[1]
        kwargs_real.update(kwargs)

        xlora_classifier = xlora_state.get_xlora_classifier()

        if "_xlora_classifier_inhibitor_flag" in kwargs_real:
            assert isinstance(kwargs_real["_xlora_classifier_inhibitor_flag"], int)
            batch_size = kwargs_real["_xlora_classifier_inhibitor_flag"]
            del kwargs_real["_xlora_classifier_inhibitor_flag"]
            xlora_state.set_scalings(torch.zeros(batch_size, xlora_classifier.n_layers, xlora_classifier.n_classes))
            return

        xlora_scalings = xlora_classifier.forward(
            *args_real,
            **kwargs_real,
        )
        xlora_state.set_scalings(xlora_scalings)

    model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

    adapters_items = list(adapters.items())
    first_item = adapters_items[0]
    adapters_items = adapters_items[1:]
    model_peft = PeftModel.from_pretrained(model, first_item[1], first_item[0], False)
    for adapter_name, model_id in adapters_items:
        model_peft.load_adapter(model_id, adapter_name)

    model_peft.base_model.set_adapter(list(adapters.keys()))

    assert isinstance(model_peft.base_model, peft.tuners.lora.LoraModel)

    base_model_wrapper = BaseTunerWrapper(model_peft.base_model)
    model_peft.base_model.forward = base_model_wrapper.forward  # type: ignore[method-assign]

    peft_model_wrapper = PeftModelWrapper(model_peft)
    model_peft.save_pretrained = peft_model_wrapper.save_pretrained  # type: ignore[method-assign]

    total_swapped = convert_layers_to_xlora(
        model_peft,
        verbose,
        xlora_config.top_k_lora,
    )

    n_classes = len(adapters)
    xlora_classifier = xLoRAClassifier(model_peft, xlora_config, n_classes, total_swapped)
    xlora_state.set_xlora_classifier(xlora_classifier)

    for name, param in model.base_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False

    return model_peft


def from_pretrained(
    load_directory: str,
    from_safetensors: bool,
    model: PreTrainedModel,
    xlora_config: xLoRAConfig,
    verbose: bool,
    adapters: Dict[str, str],
) -> PeftModel:
    """
    Loads a pretrained classifier from the specified folder while initializing the model. This is the counterpart to `xLoRAModel.save_pretrained`.

    This method is very similar to `add_xlora_to_model`: it converts all LoRA adapters to xLoRA layers, and it is one of
    the intended entrypoints for use of xLoRA. All LoRA adapters will be frozen, and the xLoRAClassifier is initialized.

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
    Returns:
        model (`PeftModel`):
            The new model.
    """

    model_peft = add_xlora_to_model(model, xlora_config, adapters, verbose)

    classifier = xlora_state.get_xlora_classifier()
    with open(os.path.join(load_directory, "xlora_classifier_config.json"), "w") as f:
        conf = json.load(f)
        assert classifier.n_classes == conf["n_classes"]

    if from_safetensors:
        state_dict = safetensors.torch.load_file(  # type: ignore
            os.path.join(load_directory, "xlora_classifier.safetensors"),
            device={k: v.device for k, v in classifier.state_dict()},  # type: ignore
        )
    else:
        state_dict = torch.load(os.path.join(load_directory, "xlora_classifier.pt"))
    classifier.load_state_dict(state_dict)

    return model_peft


def set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int):
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses. Following
    this, the value of the scalings will be reset to the previous value. If the original value had a lifetime,
    only the value which it would have if it were read at assignment-time will be preserved.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    xlora_state.set_scalings_lifetime(value, n_accesses_lifetime)


def print_scalings_predictions(n_predictions_lifetime: int):
    """
    Print the scaling states for the next n classifier predictions (i.e. forward, generate passes)
    """
    xlora_classifier.set_n_predictions_lifetime(n_predictions_lifetime)


def enable_scalings_logging():
    """
    Enable scalings logging.
    """
    xlora_classifier.set_scalings_logging(True)


def disable_scalings_logging():
    """
    Disable scalings logging.
    """
    xlora_classifier.set_scalings_logging(False)


def flush_log_scalings(path: str):
    """
    Write the scalings log (a tensor of shape (num_logged, batch_size, n_layers, n_classes)) to the specified path.
    """
    classfier = xlora_state.get_xlora_classifier()
    classfier.flush_log_scalings(path)


def get_nb_trainable_parameters(model: PeftModel) -> Tuple[int, int]:
    """
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    model_trainable_params, model_all_param = model.get_nb_trainable_parameters()

    xlora_classifier = xlora_state.get_xlora_classifier()
    xlora_trainable_params, xlora_all_param = xlora_classifier.get_nb_trainable_parameters()

    trainable_params, all_param = (
        (model_trainable_params + xlora_trainable_params),
        (model_all_param + xlora_all_param),
    )

    return trainable_params, all_param


def print_trainable_parameters(model: PeftModel):
    """
    Prints the number of trainable parameters in the model, including of the xLoRA classifier.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )
