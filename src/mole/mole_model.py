import collections
import os
from typing import Optional

import peft
import safetensors
import torch
import torch.nn as nn
from peft.mixed_model import PeftMixedModel

from . import mole_state


class MoLEModel(nn.Module):
    def __init__(self, model: PeftMixedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: Optional[bool] = True,
        is_main_process: bool = True,
    ) -> None:
        r"""
        This function saves the classifier weights to a directory.

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
        state_dict = classifier.state_dict()
        if safe_serialization:
            # https://github.com/huggingface/peft/blob/main/src/peft/peft_model.py#L223
            if is_main_process and safe_serialization:
                # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # Safetensors does not allow tensor aliasing.
                # We're going to remove aliases before saving
                ptrs = collections.defaultdict(list)
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

                safetensors.torch.save_file(
                    state_dict, os.path.join(save_directory, "mole_classifier.safetensors"), metadata={"format": "pt"}
                )
        elif is_main_process:
            torch.save(state_dict, os.path.join(save_directory, "mole_classifier.safetensors"))

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.model(*args, **kwargs)
