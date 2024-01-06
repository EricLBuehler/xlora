import typing
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft.mixed_model import PeftMixedModel
from transformers.modeling_outputs import BaseModelOutputWithPast  # type: ignore

from .mole_config import MoLEConfig


class MoLEClassifier(nn.Module):
    """
    A classifier to select LoRA layers for MoLE. It runs the base model with LoRA adapter scalings of 0.
    """

    def __init__(self, model: PeftMixedModel, config: MoLEConfig, n_classes: int):
        super().__init__()

        self.model = model
        self.n_classes = n_classes
        self.config = config

        self.inner: nn.ModuleList = nn.ModuleList([])
        if self.config.mole_depth == 1:
            self.inner.append(nn.Linear(config.vocab_size, n_classes, bias=False).to(config.device))
        elif self.config.mole_depth == 2:
            self.inner.append(nn.Linear(config.vocab_size, config.mole_size, bias=False).to(config.device))
            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device))
        else:
            assert self.config.mole_depth > 0
            self.inner.append(nn.Linear(config.vocab_size, config.mole_size, bias=False).to(config.device))

            for _ in range(config.mole_depth - 2):
                self.inner.append(nn.Linear(config.mole_size, config.mole_size, bias=False).to(config.device))

            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Using the input, predict `n_classes` LoRA alpha values.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = typing.cast(torch.FloatTensor, inputs_embeds).shape[0]

        result: Union[Tuple, BaseModelOutputWithPast] = self.model.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            _mole_classifier_inhibitor_flag=batch_size,
        )
        hidden_states = result[0]

        logits = hidden_states
        for layer in self.inner:
            logits = layer.forward(logits)

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
        pooled_logits: torch.Tensor = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        if self.config.enable_softmax:
            pooled_logits = pooled_logits.softmax(dim=-1)

        return pooled_logits

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
