from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # type: ignore

from .xlora import from_pretrained, xLoRAModel  # type: ignore


def load_model(
    model_name: str,
    fine_tune_model_name: Optional[str],
    device: str,
    dtype: torch.dtype,
    adapters: Union[List[str], Dict[str, str]],
    use_flash_attention_2: bool = False,
    load_xlora: bool = True,
    verbose: bool = False,
    use_cache: bool = False,
    from_safetensors: bool = True,
) -> Tuple[Union[AutoModelForCausalLM, xLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Convenience function to load a model, converting it to xLoRA if specified.

    Args:
        model_name (`str`):
            AutoModelForCausalLM pretrained model name or path
        fine_tune_model_name (`str`, *optional*):
            Directory to load the xLoRAClassifier from.
        device (`str`):
            Device to load the base model and the xLoRA model to.
        dtype (`torch.dtype`):
            Datatype for the base model.
        adapters (`list` or `dict):
            List of adapter names (the keys of the adapters `dict` in `add_xlora_to_model`) OR Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
            Specify the list if the adapters were trainable.
        use_flash_attention_2 (`bool`, *optional*, defaults to False):
            Use FlashAttention v2 for the base model.
        load_xlora (`bool`, *optional*, defaults to True):
            Load the model to xLoRA.
        verbose (`bool`, *optional*, defaults to False):
            Enable verbose loading.
        use_cache (`bool`, *optional*, defaults to False):
            If the base model config has a `use_cache` attribute, set it to this value.
        from_safetensors (`bool`, *optional*, defaults to True):
            Whether to load the classifier weights from a .pt or .safetensors file.

    Returns:
        Tuple whose elements are respectively:

        model (`AutoModelForCausalLM` or `xLoRAModel`):
            The model.

        tokenizer (`AutoTokenizer`):
            The tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=dtype,
        use_flash_attention_2=use_flash_attention_2,
        verbose=verbose,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device,
        verbose=verbose,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_xlora:
        assert fine_tune_model_name is not None
        model = from_pretrained(
            load_directory=fine_tune_model_name,
            from_safetensors=from_safetensors,
            model=model,
            adapters=adapters,
            verbose=verbose,
            device=device,
        )
        if verbose:
            print("X-LoRA loaded.")

    elif verbose:
        print("No X-LoRA loaded, just base model.")

    return model, tokenizer
