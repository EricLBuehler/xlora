from typing import Optional

import torch


class MoLEConfig:
    r"""
    This is the configuration class to store the configuration of a [`MoLEClassifier`].

    Args:
        hidden_size (`int`):
            Hidden size of the base model.
        device (`torch.device`):
            Device for the MoLE classifier.
        enable_softmax (`bool`, *optional*, defaults to `True`):
            Enable softmax application for the MoLE classifier.
        top_k_lora (`int`, *optional*, defaults to None):
            Sparsely select the top_k LoRA experts instead of the default dense method.
        mole_depth (`int`, *optional*, defaults to 1):
            Depth of the MoLE classifier.
        mole_size (`int`, *optional*, defaults to 32):
            Hidden size of the MoLE classifier, irrelevant if `mole_depth=1`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.

    """

    model_type = "mole"

    def __init__(
        self,
        hidden_size: int,
        device: torch.device,
        enable_softmax: bool = True,
        top_k_lora: Optional[int] = None,
        mole_depth: int = 1,
        mole_size: int = 32,
        pad_token_id: Optional[int] = None,
    ):
        self.device = device
        self.enable_softmax = enable_softmax
        self.top_k_lora = top_k_lora
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.mole_depth = mole_depth
        self.mole_size = mole_size
