from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class xLoRAConfig:
    r"""
    This is the configuration class to store the configuration of a [`xLoRAClassifier`].

    Args:
        hidden_size (`int`):
            Hidden size of the base model.
        device (`torch.device`):
            Device for the X-LoRA classifier.
        enable_softmax (`bool`, *optional*, defaults to `True`):
            Enable softmax application for the X-LoRA classifier.
        layerwise_scalings (`bool`, *optional*, defaults to `False`):
            Generate scalings for each layer.
        top_k_lora (`int`, *optional*, defaults to None):
            Sparsely select the top_k LoRA experts instead of the default dense method.
        xlora_depth (`int`, *optional*, defaults to 1):
            Depth of the X-LoRA classifier.
        xlora_size (`int`, *optional*, defaults to 32):
            Hidden size of the X-LoRA classifier, irrelevant if `xlora_depth=1`.
        enable_relu_and_dropout (`bool`, *optional*, defaults to `False`):
            Enable ReLU activation and Dropout application of the X-LoRA classifier.
        use_bias (`bool`, *optional*, defaults to `True`):
            Enable bias in X-LoRA classifier.
        xlora_dropout_p (`float`, *optional*, defaults to 0.5):
            Dropout probability of the X-LoRA classifier, irrelevant if `xlora_depth=1` or `enable_relu_and_dropout=True`.
        stop_token_id (`int`, *optional*):
            The id of the stop token for the input. If this is None, the sequence length is calculated using the attention mask.
        use_trainable_adapters (`bool`, *optional`, defaults to False):
            Make the adapters trainable.
        use_mean_pool (`bool`, *optional*, defaults to True):
            Sum the hidden states for the last token
    """

    model_type = "xlora"

    hidden_size: int
    device: torch.device
    enable_softmax: bool = True
    layerwise_scalings: bool = False
    top_k_lora: Optional[int] = None
    xlora_depth: int = 1
    xlora_size: int = 2048
    enable_relu_and_dropout: bool = False
    use_bias: bool = True
    xlora_dropout_p: float = 0.2
    stop_token_id: Optional[int] = None
    use_trainable_adapters: bool = False
    use_mean_pool: bool = False  # TODO(all): test
