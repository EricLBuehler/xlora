from typing import Dict, List, Optional

import torch.nn as nn
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig
from transformers.configuration_utils import PretrainedConfig

from mole.mole import mole_state
from mole.mole.mole_classifier import MoLEClassifier
from mole.mole.mole_insertion_layers import MoLELayer


class MoLEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoLEClassifier`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        top_k_lora (`int`, *optional*, defaults to None):
            Sparesely select the top_k LoRA experts instead of the default dense method.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mistral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    """

    model_type = "mistral"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        top_k_lora=None,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.top_k_lora = top_k_lora

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def convert_layers_to_mole(
    base: nn.Module,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
    top_k_lora: Optional[int] = None,
):
    modules = list(base.modules())
    for module in modules:
        if isinstance(module, lora.LoraLayer):
            new_layer = MoLELayer(
                adapters=adapters,
                target=module,
                peft_config=peft_config,
                combination_type=combination_type,
                svd_rank=svd_rank,
                svd_clamp=svd_clamp,
                svd_full_matrices=svd_full_matrices,
                svd_driver=svd_driver,
                top_k_lora=top_k_lora,
            )
            module.forward = new_layer.forward


def add_mole_to_model(
    model: nn.Module,
    mole_config: MoLEConfig,
    adapters: List[str],
    peft_config: Dict[str, PeftConfig],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
):
    """
    This method converts all LoRA adapters to MoLE layers, and it is the intended entrypoint
    for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

    When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
    the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
    errors.

    Args:
        model (`Module`):
            The model to recursively loop over to find and convert all LoRA adapters.
        adapters (`list`):
            List of adapter names to use as LoRA experts.
        peft_config: (`dict`):
            PeftConfigs for each adapter in the LoraLayer.
        combination_type (`str`):
            Type of merging. Can be one of [`svd`, `linear`, `cat`]. When using the `cat` combination_type you
            should be aware that rank of the resulting adapter will be equal to the sum of all adapters ranks. So
            it's possible that the mixed adapter may become too big and result in OOM errors.
        svd_rank (`int`, *optional*):
            Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
        svd_clamp (`float`, *optional*):
            A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
            clamping. Defaults to None.
        svd_full_matrices (`bool`, *optional*):
            Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
            tensors U and Vh. Defaults to True.
        svd_driver (`str`, *optional*):
            Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
            one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
            documentation. Defaults to None.
    """
    convert_layers_to_mole(
        model,
        adapters,
        peft_config,
        combination_type,
        svd_rank,
        svd_clamp,
        svd_full_matrices,
        svd_driver,
        top_k_lora=mole_config.top_k_lora,
    )

    n_classes = len(adapters)
    mole_classifier = MoLEClassifier(mole_config, n_classes)

    def hook(module, *args, **kwargs) -> None:
        mole_output = mole_classifier.forward(
            *args,
            **kwargs,
        )
        mole_scalings = list(mole_output)
        mole_state.set_scalings(mole_scalings)

    model.register_forward_pre_hook(hook)
