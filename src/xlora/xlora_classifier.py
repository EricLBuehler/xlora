import typing
from typing import List, Optional, Union

import numpy
import torch
import torch.nn as nn
from peft.peft_model import PeftModel
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput  # type: ignore

from .xlora_config import xLoRAConfig

_n_predictions_lifetime: int = 0
_scalings_logging: bool = False


def get_n_predictions_lifetime() -> int:
    global _n_predictions_lifetime
    """
    Reads the n predictions lifetime.
    """
    assert _n_predictions_lifetime is not None
    return _n_predictions_lifetime


def set_n_predictions_lifetime(value: int) -> None:
    global _n_predictions_lifetime
    """
    Sets the n predictions lifetime.
    """
    _n_predictions_lifetime = value


def set_scalings_logging(value: bool):
    global _scalings_logging
    _scalings_logging = value


class xLoRAClassifier(nn.Module):
    """
    A classifier to select LoRA layers for xLoRA. It runs the base model with LoRA adapter scalings of 0.
    """

    def __init__(
        self,
        model: PeftModel,
        config: xLoRAConfig,
        n_classes: int,
        n_layers: int,
    ):
        super().__init__()

        # To avoid registering this with nn.Module
        self.__dict__["model"] = model
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings: List[torch.Tensor] = []

        dtype = next(model.parameters()).dtype

        self.inner: nn.ModuleList = nn.ModuleList([])
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:
                self.last = nn.Linear(config.hidden_size, n_classes * n_layers, bias=False).to(config.device).to(dtype)
            else:
                self.last = nn.Linear(config.hidden_size, n_classes, bias=False).to(config.device).to(dtype)
        elif self.config.xlora_depth == 2:
            self.inner.append(nn.Linear(config.hidden_size, config.xlora_size, bias=False).to(config.device).to(dtype))
            if config.layerwise_scalings:
                self.last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=False).to(config.device).to(dtype)
            else:
                self.last = nn.Linear(config.xlora_size, n_classes, bias=False).to(config.device).to(dtype)
        else:
            assert self.config.xlora_depth > 0
            self.inner.append(nn.Linear(config.hidden_size, config.xlora_size, bias=False).to(config.device).to(dtype))

            for _ in range(config.xlora_depth - 2):
                self.inner.append(
                    nn.Linear(config.xlora_size, config.xlora_size, bias=False).to(config.device).to(dtype)
                )

            if config.layerwise_scalings:
                self.last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=False).to(config.device).to(dtype)
            else:
                self.last = nn.Linear(config.xlora_size, n_classes, bias=False).to(config.device).to(dtype)

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

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = typing.cast(torch.FloatTensor, inputs_embeds).shape[1]

        # For type checking
        model: PeftModel = self.model  # type: ignore
        with model.disable_adapter():
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            result: ModelOutput = model.forward(
                *args,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                _xlora_classifier_inhibitor_flag=batch_size,
                **kwargs,
            )

            assert isinstance(result, tuple) or isinstance(result, CausalLMOutputWithPast)

        if isinstance(result, tuple):
            hidden_states = result[3]
        else:
            hidden_states = result.hidden_states

        assert hidden_states is not None

        hidden_state = hidden_states[-1]  # Get the last hidden state

        """
        # ModelOutput is the superclass, really this is a @dataclass instance and must have `.hidden_states`. If it does not,
        # the model cannot be used.
        hidden_states: List[torch.FloatTensor] = result.hidden_states  # type:ignore

        assert hidden_states is not None

        hidden_state = hidden_states[-1]  # Get the last hidden state
        """

        for layer in self.inner:
            hidden_state = layer.forward(hidden_state)

        logits = self.last.forward(hidden_state)
        if not self.config.layerwise_scalings:
            logits = logits.repeat(1, 1, self.n_layers)
        logits = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)

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

        if _scalings_logging:
            self.log_scalings.append(scalings.unsqueeze(0))

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

    def flush_log_scalings(self, path: str):
        if not _scalings_logging:
            raise Exception("Scalings logging is disabled!")

        if len(self.log_scalings) == 0:
            raise ValueError("No log scalings to flush.")

        result = torch.cat(self.log_scalings, dim=0)
        npy = result.numpy()
        numpy.save(path, npy)
