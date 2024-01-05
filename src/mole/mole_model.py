import collections
import json
import os
from typing import Optional, Tuple, Union

import peft
import safetensors
import torch
import torch.nn as nn
from peft.mixed_model import PeftMixedModel

from . import mole_state


class MoLEModel(nn.Module):
    """
    This class is the intended way to interact with a model that has been converted to MoLE.
    """

    def __init__(self, model: PeftMixedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = model

    @staticmethod
    def _calculate_weights(*args, **kwargs):
        mole_classifier = mole_state.get_mole_classifier()

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

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        self._calculate_weights(*args, **kwargs)
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """
        Generate output.
        """
        self._calculate_weights(*args, **kwargs)
        return self.model.generate(*args, **kwargs)

    def save_pretrained(
        self,
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
            torch.save(state_dict, os.path.join(save_directory, "mole_classifier.pt"))

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        """Moves all model and MoLE classifier parameters and buffers to the GPU."""
        self.model = self.model.cuda(device=device)
        classifier = mole_state.get_mole_classifier()
        classifier = classifier.cuda(device=device)
        mole_state.set_mole_classifier(classifier)

    def ipu(self, device: Optional[Union[int, torch.device]] = None):
        """Moves all model and MoLE classifier parameters and buffers to the IPU."""
        self.model = self.model.ipu(device=device)
        classifier = mole_state.get_mole_classifier()
        classifier = classifier.ipu(device=device)
        mole_state.set_mole_classifier(classifier)

    def xpu(self, device: Optional[Union[int, torch.device]] = None):
        """Moves all model and MoLE classifier parameters and buffers to the XPU."""
        self.model = self.model.xpu(device=device)
        classifier = mole_state.get_mole_classifier()
        classifier = classifier.xpu(device=device)
        mole_state.set_mole_classifier(classifier)

    def cpu(self):
        """Moves all model and MoLE classifier parameters and buffers to the CPU. Modifies the models in place."""
        self.model.cpu()
        classifier = mole_state.get_mole_classifier()
        classifier.cpu()

    def type(self, dst_type: Union[torch.dtype, str]):
        """Casts all parameters and buffers of the model and MoLE classifier to `dst_type`. Modifies the models in place."""
        self.model.type(dst_type=dst_type)
        classifier = mole_state.get_mole_classifier()
        classifier.type(dst_type=dst_type)

    def eval(self):
        """Sets the model and MoLE classifier in evaluation mode."""
        self.model.eval()
        classifier = mole_state.get_mole_classifier()
        classifier.eval()

    def train(self, mode: Optional[bool] = True):
        """Sets the model and MoLE classifier in training mode if `mode=True`, evaluation mode if `mode=False`."""
        self.model.train(mode=mode)
        classifier = mole_state.get_mole_classifier()
        classifier.train(mode=mode)

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        """
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        model_trainable_params, model_all_param = self.model.get_nb_trainable_parameters()

        mole_classifier = mole_state.get_mole_classifier()
        mole_trainable_params, mole_all_param = mole_classifier.get_nb_trainable_parameters()

        trainable_params, all_param = (
            (model_trainable_params + mole_trainable_params),
            (model_all_param + mole_all_param),
        )

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model, including of the MoLE classifier.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )
