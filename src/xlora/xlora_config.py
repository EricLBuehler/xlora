import warnings
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
        enable_softmax_topk (`bool`, *optional*, defaults to `False`):
            Enable softmax application for the top-k LoRA adapters. Mutually exclusive to `enable_softmax`.
        softmax_temperature (`float`): softmax temperature, lower yields sharper predictions
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
    enable_softmax_topk: bool = False
    layerwise_scalings: bool = False
    xlora_depth: int = 1
    xlora_size: int = 2048
    enable_relu_and_dropout: bool = False
    use_bias: bool = True
    xlora_dropout_p: float = 0.2
    stop_token_id: Optional[int] = None
    use_trainable_adapters: bool = False
    use_mean_pool: bool = False  # TODO(all): test. See #9
    softmax_temperature: float = 1.0
    top_k_lora: Optional[int] = None

    def __post_init__(self):
        if self.enable_softmax_topk and self.enable_softmax:
            warnings.warn(
                "`enable_softmax_topk` and `enable_softmax` are both enabled. This will result in worse performance."
            )

        if self.use_mean_pool:
            warnings.warn("`use_mean_pool` implementation is currently in testing. See #9.")

        if self.top_k_lora is not None and self.top_k_lora < 1:
            warnings.warn("`top_k_lora` value must be at least 1.")
