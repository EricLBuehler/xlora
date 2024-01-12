import operator
import typing
from dataclasses import replace
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner, PeftConfig  # type: ignore
from torch import Tensor

from mole import mole_state

MOLE_ADAPTER_NAME = "mole_adapter"


# https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L266
def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class MoLEBaseLayer:
    """
    This is a utility class which manages merging LoRA adapters using specified weights.
    """

    prefix = lora.LoraModel.prefix

    @classmethod
    def add_weighted_adapter(
        cls,
        model: BaseTuner,
        target: lora.LoraLayer,
        adapters: List[str],
        weights: List[Tensor],
        adapter_name: str,
        peft_config: Dict[str, lora.LoraConfig],
        combination_type: str = "cat",
    ) -> None:
        for adapter in adapters:
            if adapter not in list(peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [peft_config[adapter].r for adapter in adapters]  # type: ignore
        if combination_type == "linear":
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] == str:
            new_target_modules: Union[str, List[str], None] = "|".join(
                f"({peft_config[adapter].target_modules})" for adapter in adapters
            )
        elif target_module_types[0] == set:
            new_target_modules = reduce(operator.or_, (peft_config[adapter].target_modules for adapter in adapters))
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        peft_config[adapter_name] = replace(
            peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
        )
        model.inject_adapter(model, adapter_name)

        key_list = [key for key, _ in model.named_modules() if cls.prefix not in key]
        for key in key_list:
            _, target, _ = typing.cast(Tuple[Any, lora.LoraLayer, Any], _get_submodules(model, key))
            if isinstance(target, lora.LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0  # type: ignore
                target_lora_B.data = target_lora_B.data * 0.0  # type: ignore
                if combination_type == "linear":
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        target_lora_A.data += current_adapter_lora_A.data * weight.sqrt() * target.scaling[adapter]  # type: ignore
                        target_lora_B.data += current_adapter_lora_B.data * weight.sqrt()  # type: ignore
                elif combination_type == "cat":
                    loras_A: List[Tensor] = []
                    loras_B: List[Tensor] = []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(typing.cast(Tensor, current_adapter_lora_B.data))

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
                    loras_A_cat: Tensor = torch.cat(loras_A, dim=0)
                    loras_B_cat: Tensor = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A_cat.shape[0], :] = loras_A_cat
                    target_lora_B.data[:, : loras_B_cat.shape[1]] = loras_B_cat


class MoLELayer(MoLEBaseLayer):
    """
    A MoLELayer wraps any LoraLayer and performs the MoLE operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from mole_state to execute the
    MoLE algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        adapters: List[str],
        model: BaseTuner,
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        peft_config: Dict[str, PeftConfig],
        combination_type: str = "cat",
        top_k_lora: Optional[int] = None,
    ) -> None:
        self.adapters = adapters
        self.model = model
        self.target_forward = target_forward
        self.target = target
        self.peft_config = peft_config
        self.top_k_lora = top_k_lora

        self.combination_type = combination_type

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the MoLELayer class).
        """
        scalings = mole_state.get_scalings()
        outputs: List[Tensor] = []
        if self.top_k_lora is None:
            for batch_scalings, batch_x in zip(scalings, x):
                self.add_weighted_adapter(
                    target=self.target,
                    adapters=self.adapters,
                    model=self.model,
                    weights=list(batch_scalings),
                    adapter_name=MOLE_ADAPTER_NAME,
                    peft_config=typing.cast(Dict[str, lora.LoraConfig], self.peft_config),
                    combination_type=self.combination_type,
                )

                self.target.set_adapter(MOLE_ADAPTER_NAME)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)
        else:
            for batch_scalings, batch_x in zip(scalings, x):
                (topk_scalings, indices) = torch.topk(input=batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.adapters[i] for i in indices]
                self.add_weighted_adapter(
                    target=self.target,
                    adapters=adapters,
                    model=self.model,
                    weights=list(topk_scalings),
                    adapter_name=MOLE_ADAPTER_NAME,
                    peft_config=typing.cast(Dict[str, lora.LoraConfig], self.peft_config),
                    combination_type=self.combination_type,
                )

                self.target.set_adapter(MOLE_ADAPTER_NAME)

                output = self.target_forward(batch_x, *args, **kwargs)
                outputs.append(output)

        return torch.cat(outputs, dim=0)


class BaseTunerWrapper:
    def __init__(self, base_model: BaseTuner):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
