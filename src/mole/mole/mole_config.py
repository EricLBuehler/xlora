from typing import Optional


class MoLEConfig:
    r"""
    This is the configuration class to store the configuration of a [`MoLEClassifier`].

    Args:
        hidden_size (`int`):
            Dimension of the hidden representations for the base model.
        top_k_lora (`int`, *optional*, defaults to None):
            Sparesely select the top_k LoRA experts instead of the default dense method.
        pad_token_id (`int`, *optional*):
            The id of the padding token.

    """

    model_type = "mole"

    def __init__(
        self,
        hidden_size: int,
        top_k_lora: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.top_k_lora = top_k_lora

        self.hidden_size = hidden_size

        self.pad_token_id = pad_token_id
