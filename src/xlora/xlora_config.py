from typing import Optional

import torch


class xLoRAConfig:
    r"""
    This is the configuration class to store the configuration of a [`xLoRAClassifier`].

    Args:
        hidden_size (`int`):
            Hidden size of the base model.
        device (`torch.device`):
            Device for the xLoRA classifier.
        enable_softmax (`bool`, *optional*, defaults to `True`):
            Enable softmax application for the xLoRA classifier.
        layerwise_scalings (`bool`, *optional*, defaults to `False`):
            Generate scalings for each layer.
        top_k_lora (`int`, *optional*, defaults to None):
            Sparsely select the top_k LoRA experts instead of the default dense method.
        xlora_depth (`int`, *optional*, defaults to 1):
            Depth of the xLoRA classifier.
        xlora_size (`int`, *optional*, defaults to 32):
            Hidden size of the xLoRA classifier, irrelevant if `xlora_depth=1`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.

    """

    model_type = "xlora"

    def __init__(
        self,
        hidden_size: int,
        device: torch.device,
        enable_softmax: bool = True,
        layerwise_scalings: bool = False,
        top_k_lora: Optional[int] = None,
        xlora_depth: int = 1,
        xlora_size: int = 32,
        pad_token_id: Optional[int] = None,
    ):
        self.device = device
        self.enable_softmax = enable_softmax
        self.top_k_lora = top_k_lora
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.xlora_depth = xlora_depth
        self.xlora_size = xlora_size
        self.layerwise_scalings = layerwise_scalings
