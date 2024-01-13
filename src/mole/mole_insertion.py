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

from mole import mole_state


class MoLELayer:
    """
    A MoLELayer wraps any LoraLayer and performs the MoLE operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from mole_state to execute the
    MoLE algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        scaling_keys: List[str],
        top_k_lora: Optional[int] = None,
    ) -> None:
        self.target_forward = target_forward
        self.target = target
        self.scaling_keys = scaling_keys
        self.top_k_lora = top_k_lora

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the MoLELayer class).
        """
        old_scalings = self.target.scaling.copy()

        outputs: List[Tensor] = []
        if self.top_k_lora is None:
            for i, (batch_x, batch_scalings) in enumerate(zip(x, mole_state.get_scalings())):
                self.scale_adapters(self.target, batch_scalings, self.scaling_keys)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)

                self.target.scaling = old_scalings
        else:
            for i, (batch_x, batch_scalings) in enumerate(zip(x, mole_state.get_scalings())):
                (topk_scalings, indices) = torch.topk(input=batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.scaling_keys[i] for i in indices]

                self.scale_adapters(self.target, topk_scalings, adapters)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)

                self.target.scaling = old_scalings

        return torch.cat(outputs, dim=0)

    @staticmethod
    def scale_adapters(target: lora.LoraLayer, scalings: Tensor, adapters: List[str]):
        for scaling, adapter in zip(scalings, adapters):
            target.scaling[adapter] = target.scaling[adapter] * scaling


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class PeftModelWrapper:
    def __init__(self, base_model: PeftModel):
        self.model = base_model

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
