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
        top_k_lora: Optional[int] = None,
    ) -> None:
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
            for batch_x, batch_scalings in zip(x, xlora_state.get_scalings()):
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
            for batch_x, batch_scalings in zip(x, xlora_state.get_scalings()):
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
