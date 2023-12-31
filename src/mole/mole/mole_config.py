class MoLEConfig:
    r"""
    This is the configuration class to store the configuration of a [`MoLEClassifier`].

    Args:
        top_k_lora (`int`, *optional*, defaults to None):
            Sparesely select the top_k LoRA experts instead of the default dense method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        pad_token_id (`int`, *optional*):
            The id of the padding token.

    """

    model_type = "mole"

    def __init__(
        self,
        top_k_lora=None,
        hidden_size=4096,
        pad_token_id=None,
    ):
        self.top_k_lora = top_k_lora

        self.hidden_size = hidden_size

        self.pad_token_id = pad_token_id
