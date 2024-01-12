import operator
import typing
from dataclasses import replace
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from peft.tuners import lora
from peft.tuners.tuners_utils import BaseTuner, PeftConfig  # type: ignore
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore

from .mole_config import MoLEConfig

_MOLE_ADAPTER_NAME = "mole_adapter"


# https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L266
def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


class MoLEClassifier(nn.Module):
    """
    A classifier to select LoRA layers for MoLE. It runs the base model with LoRA adapter scalings of 0.
    """

    prefix = lora.LoraModel.prefix

    def __init__(
        self,
        model: PeftModel,
        config: MoLEConfig,
        n_classes: int,
        adapters: List[str],
        peft_config: Dict[str, PeftConfig],
        combination_type: str = "cat",
        top_k_lora: Optional[int] = None,
    ):
        super().__init__()

        self.model = model
        self.n_classes = n_classes
        self.config = config
        self.adapters = adapters
        self.combination_type = combination_type
        self.top_k_lora = top_k_lora
        self.peft_config = peft_config

        self.inner: nn.ModuleList = nn.ModuleList([])
        if self.config.mole_depth == 1:
            self.inner.append(nn.Linear(config.hidden_size, n_classes, bias=False).to(config.device))
        elif self.config.mole_depth == 2:
            self.inner.append(nn.Linear(config.hidden_size, config.mole_size, bias=False).to(config.device))
            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device))
        else:
            assert self.config.mole_depth > 0
            self.inner.append(nn.Linear(config.hidden_size, config.mole_size, bias=False).to(config.device))

            for _ in range(config.mole_depth - 2):
                self.inner.append(nn.Linear(config.mole_size, config.mole_size, bias=False).to(config.device))

            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = typing.cast(torch.FloatTensor, inputs_embeds).shape[0]

        with self.model.disable_adapter():
            kwargs["output_hidden_states"] = True
            result: Union[Tuple, CausalLMOutputWithPast] = self.model.forward(
                *args,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                _mole_classifier_inhibitor_flag=batch_size,
                **kwargs,
            )

            assert isinstance(result, tuple) or isinstance(result, CausalLMOutputWithPast)

        if isinstance(result, tuple):
            hidden_states = result[3]
        else:
            hidden_states = result.hidden_states

        assert hidden_states is not None

        hidden_state = hidden_states[-1]  # Get the last hidden state

        for layer in self.inner:
            hidden_state = layer.forward(hidden_state)

        logits = hidden_state

        if self.config.pad_token_id is None:
            sequence_lengths: Union[int, torch.Tensor] = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        # Get it for the last token
        scalings: torch.Tensor = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        if self.config.enable_softmax:
            scalings = scalings.softmax(dim=-1)

        assert isinstance(self.model, BaseTuner)
        if self.top_k_lora is None:
            for i, batch_scalings in enumerate(scalings):
                self.add_weighted_adapter(
                    adapters=self.adapters,
                    model=self.model,
                    weights=list(batch_scalings),
                    adapter_name=_MOLE_ADAPTER_NAME + f"_{i}",
                    peft_config=typing.cast(Dict[str, lora.LoraConfig], self.peft_config),
                    combination_type=self.combination_type,
                )
        else:
            for i, batch_scalings in enumerate(scalings):
                (topk_scalings, indices) = torch.topk(input=batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.adapters[i] for i in indices]
                self.add_weighted_adapter(
                    adapters=adapters,
                    model=self.model,
                    weights=list(topk_scalings),
                    adapter_name=_MOLE_ADAPTER_NAME + f"_{i}",
                    peft_config=typing.cast(Dict[str, lora.LoraConfig], self.peft_config),
                    combination_type=self.combination_type,
                )

        return scalings

    def get_nb_trainable_parameters(self):
        # https://github.com/huggingface/peft/blob/main/src/peft/mixed_model.py#L156
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel  # type: ignore

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    @classmethod
    def add_weighted_adapter(
        cls,
        model: BaseTuner,
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
