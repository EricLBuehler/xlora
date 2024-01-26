import collections
import inspect
import json
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import peft
import safetensors  # type: ignore
import torch
from peft.peft_model import PeftModel
from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner  # type: ignore
from torch import Tensor

from xlora.xlora_classifier import xLoRAClassifier
from xlora.xlora_config import xLoRAConfig


class xLoRALayer:
    """
    A xLoRALayer wraps any LoraLayer and performs the xLoRA operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from xlora_state to execute the
    xLoRA algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        model: PeftModel,
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        layer_number: int,
        top_k_lora: Optional[int] = None,
    ) -> None:
        self.model = model
        self.target_forward = target_forward
        self.target = target
        self.scaling_keys = scaling_keys
        self.top_k_lora = top_k_lora
        self.layer_number = layer_number
        self.disabled = False

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        outputs: List[Tensor] = []
        if self.top_k_lora is None:
            for batch_x, batch_scalings in zip(x, self.model.internal_xlora_scalings):  # type: ignore
                layer_batch_scalings = batch_scalings[self.layer_number]
                if not self.disabled:
                    self.scale_adapters(self.target, layer_batch_scalings, self.scaling_keys)
                    output = self.target_forward(batch_x.unsqueeze(dim=0), *args, **kwargs)
                    outputs.append(output)
                    self.unscale_adapters(self.target, layer_batch_scalings, self.scaling_keys)
                else:  # If disabled just run the model w/o adapters and w/o scaling NOTE(EricLBuehler): not implemented
                    output = self.target_forward(batch_x.unsqueeze(dim=0), *args, **kwargs)
                    outputs.append(output)
        else:
            for batch_x, batch_scalings in zip(x, self.model.internal_xlora_scalings):  # type: ignore
                layer_batch_scalings = batch_scalings[self.layer_number]

                (topk_scalings, indices) = torch.topk(input=layer_batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.scaling_keys[i] for i in indices]

                if not self.disabled:
                    self.scale_adapters(self.target, topk_scalings, adapters)
                    output = self.target_forward(batch_x.unsqueeze(dim=0), *args, **kwargs)
                    outputs.append(output)
                    self.unscale_adapters(self.target, topk_scalings, adapters)
                else:  # If disabled just run the model w/o adapters and w/o scaling NOTE(EricLBuehler): not implemented
                    output = self.target_forward(batch_x.unsqueeze(dim=0), *args, **kwargs)
                    outputs.append(output)

        result = torch.cat(outputs, dim=0)
        return result

    @staticmethod
    def scale_adapters(target: lora.LoraLayer, scalings: Tensor, adapters: List[str]):
        for scaling, adapter in zip(scalings, adapters):
            target.scaling[adapter] = target.scaling[adapter] * scaling

    @staticmethod
    def unscale_adapters(target: lora.LoraLayer, scalings: Tensor, adapters: List[str]):
        for scaling, adapter in zip(scalings, adapters):
            target.scaling[adapter] = target.scaling[adapter] / scaling


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner, classifier: xLoRAClassifier):
        self.model = base_model.model
        self.classifier = classifier

    def forward(self, *args, **kwargs):
        if "_xlora_classifier_inhibitor_flag" not in kwargs:
            self.classifier.forward(*args, **kwargs)
        else:
            del kwargs["_xlora_classifier_inhibitor_flag"]
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs_modified = kwargs.copy()
        kwargs_modified = {
            k: v for k, v in kwargs_modified.items() if k in inspect.signature(self.model.forward).parameters
        }
        self.classifier.forward(*args, **kwargs_modified)
        return self.model.generate(*args, **kwargs)


class PeftModelWrapper:
    def __init__(
        self,
        base_model: PeftModel,
        base_model_save: Callable[..., None],
        config: xLoRAConfig,
        base_model_get_nb_trainable_parameters: Callable[..., Tuple[int, int]],
    ):
        self.model = base_model
        self.base_model_save = base_model_save
        self.config = config
        self.base_model_get_nb_trainable_parameters = base_model_get_nb_trainable_parameters

    def print_scalings_predictions(self, n_predictions_lifetime: int):
        """
        Print the scaling states for the next n classifier predictions (i.e. forward, generate passes)
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.n_predictions_lifetime = n_predictions_lifetime

    def enable_scalings_logging(self):
        """
        Enable scalings logging.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = True

    def disable_scalings_logging(self):
        """
        Disable scalings logging, clearing the log.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = False
        classifier.log_scalings = []

    def flush_log_scalings(self, path: str):
        """
        Write the scalings log (a tensor of shape (num_logged, batch_size, n_layers, n_classes)) to the specified path.
        """
        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.flush_log_scalings(path)

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        model_trainable_params, model_all_param = self.model.get_nb_trainable_parameters()

        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore
        xlora_trainable_params, xlora_all_param = classifier.get_nb_trainable_parameters()

        trainable_params, all_param = (
            (model_trainable_params + xlora_trainable_params),
            (model_all_param + xlora_all_param),
        )

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model, including of the xLoRA classifier.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def set_use_trainable_adapters(self, use_trainable_adapters: bool):
        """
        Set the adapters to trainable or not trainable.
        """
        for name, param in self.model.base_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = use_trainable_adapters

        self.config.use_trainable_adapters = use_trainable_adapters

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
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

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)

        classifier: xLoRAClassifier = self.model.internal_xlora_classifier  # type: ignore

        conf = classifier.config.__dict__.copy()
        del conf["device"]
        with open(os.path.join(save_directory, "xlora_config.json"), "w") as f:
            json.dump(conf, f)

        if self.config.use_trainable_adapters:
            if is_main_process:
                os.makedirs(os.path.join(save_directory, "adapters"), exist_ok=True)
            self.base_model_save(
                save_directory=os.path.join(save_directory, "adapters"),
                safe_serialization=safe_serialization,
                is_main_process=is_main_process,
                selected_adapters=selected_adapters,
                save_embedding_layers=save_embedding_layers,
                **kwargs,
            )

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
                    state_dict, os.path.join(save_directory, "xlora_classifier.safetensors"), metadata={"format": "pt"}
                )
        elif is_main_process:
            torch.save(state_dict, os.path.join(save_directory, "xlora_classifier.pt"))
