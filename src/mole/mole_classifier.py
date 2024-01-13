import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from transformers.modeling_outputs import CausalLMOutputWithPast  # type: ignore

from mole.mole_state import get_n_predictions_lifetime, set_n_predictions_lifetime  # type: ignore

from .mole_config import MoLEConfig


class MoLEClassifier(nn.Module):
    """
    A classifier to select LoRA layers for MoLE. It runs the base model with LoRA adapter scalings of 0.
    """

    def __init__(
        self,
        model: PeftModel,
        config: MoLEConfig,
        n_classes: int,
    ):
        super().__init__()

        self.model = model
        self.n_classes = n_classes
        self.config = config

        dtype = next(model.parameters()).dtype

        self.inner: nn.ModuleList = nn.ModuleList([])
        if self.config.mole_depth == 1:
            self.inner.append(nn.Linear(config.hidden_size, n_classes, bias=False).to(config.device).to(dtype))
        elif self.config.mole_depth == 2:
            self.inner.append(nn.Linear(config.hidden_size, config.mole_size, bias=False).to(config.device).to(dtype))
            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device).to(dtype))
        else:
            assert self.config.mole_depth > 0
            self.inner.append(nn.Linear(config.hidden_size, config.mole_size, bias=False).to(config.device).to(dtype))

            for _ in range(config.mole_depth - 2):
                self.inner.append(
                    nn.Linear(config.mole_size, config.mole_size, bias=False).to(config.device).to(dtype)
                )

            self.inner.append(nn.Linear(config.mole_size, n_classes, bias=False).to(config.device).to(dtype))

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

        n_pred_life = get_n_predictions_lifetime()
        if n_pred_life > 0:
            print(f"Scaling predictions: {scalings}")
            set_n_predictions_lifetime(n_pred_life - 1)

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
