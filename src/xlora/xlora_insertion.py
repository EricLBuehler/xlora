import collections
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

from xlora.xlora_config import xLoRAConfig

from . import xlora_state


class xLoRALayer:
    """
    A xLoRALayer wraps any LoraLayer and performs the xLoRA operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from xlora_state to execute the
    xLoRA algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        layer_number: int,
    ) -> None:
        self.target_forward = target_forward
        self.target = target
        self.scaling_keys = scaling_keys
        self.layer_number = layer_number
        self.disabled = False

    @staticmethod
    def apply_scalings_to_x(x: torch.Tensor, scalings: torch.Tensor, adapter: int, layer: int) -> torch.Tensor:
        scalings = scalings[:, layer, adapter].unsqueeze(1).unsqueeze(1)
        return x * scalings


class xLoRALinearLayer(xLoRALayer):
    def __init__(
        self,
        target: lora.Linear,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        layer_number: int,
    ) -> None:
        super().__init__(target, target_forward, scaling_keys, layer_number)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = xlora_state.get_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x = self.apply_scalings_to_x(x, xlora_scalings, adapter_n, self.layer_number)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result


class xLoRAEmbeddingLayer(xLoRALayer):
    def __init__(
        self,
        target: lora.Embedding,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        layer_number: int,
    ) -> None:
        super().__init__(target, target_forward, scaling_keys, layer_number)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        xlora_scalings = xlora_state.get_scalings()

        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_embedding_A:
                    continue
                embedding_A = self.target.lora_embedding_A[active_adapter].T
                embedding_B = self.target.lora_embedding_B[active_adapter].T
                scaling = self.target.scaling[active_adapter]
                x = self.apply_scalings_to_x(x, xlora_scalings, adapter_n, self.layer_number)
                after_A = self.target._embed(x, embedding_A)  # type: ignore
                result += (after_A @ embedding_B) * scaling

        return result


class xLoRAConv2dLayer(xLoRALayer):
    def __init__(
        self,
        target: lora.Conv2d,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        layer_number: int,
    ) -> None:
        super().__init__(target, target_forward, scaling_keys, layer_number)

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the xLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = xlora_state.get_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x = self.apply_scalings_to_x(x, xlora_scalings, adapter_n, self.layer_number)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PeftModelWrapper:
    def __init__(
        self,
        base_model: PeftModel,
        base_model_save: Callable[..., None],
        config: xLoRAConfig,
    ):
        self.model = base_model
        self.base_model_save = base_model_save
        self.config = config

    def set_use_trainable_adapters(self, use_trainable_adapters: bool):
        """
        Set the adapters to trainable or not trainable.
        """
        if not use_trainable_adapters:
            self.model.base_model.eval()
            for name, param in self.model.base_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False

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

        classifier = xlora_state.get_xlora_classifier()

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
