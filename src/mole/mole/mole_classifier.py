import logging
from typing import List, Optional

import torch
import torch.nn as nn
from ..mistral import MistralDecoderLayer, MistralPreTrainedModel, MistralRMSNorm

from transformers import MistralConfig

logger = logging.get_logger(__name__)


class MoleClassifier(MistralPreTrainedModel):
    """
    A classifier to select LoRA layers for MoLE. It uses a single Mistral decoder layer to generate the LoRA alpha values.
    """

    def __init__(self, config: MistralConfig, n_classes: int):
        super().__init__(config)
        self.n_classes = n_classes

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MistralDecoderLayer(config, 0)])
        self._attn_implementation = config._attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.score = nn.Linear(config.hidden_size, n_classes, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        batch_size: int,
        input_ids: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Using the primary input, hidden states, predict `self.n_classes` LoRA alpha values.
        """
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        logits = self.score(hidden_states)

        if self.config.pad_token_id is None:
            sequence_lengths = -1
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

        return pooled_logits
