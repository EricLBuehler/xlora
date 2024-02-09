import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import huggingface_hub  # type: ignore
import numpy
import torch
import tqdm  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast  # type: ignore

from .xlora import from_pretrained, xLoRAModel  # type: ignore


def _get_file_path(
    load_directory: str,
    name: str,
) -> str:
    if os.path.exists(os.path.join(load_directory, name)):
        return os.path.join(load_directory, name)
    return huggingface_hub.hf_hub_download(load_directory, filename=name)


def _get_file_path_dir(load_directory: str, name: str, dir: str) -> str:
    if os.path.exists(os.path.join(load_directory, dir, name)):
        return os.path.join(load_directory, dir, name)
    return huggingface_hub.hf_hub_download(load_directory, filename=name, subfolder=dir)


def load_model(
    model_name: str,
    xlora_path: Optional[str],
    device: str,
    dtype: torch.dtype,
    use_flash_attention_2: bool = False,
    load_xlora: bool = True,
    verbose: bool = False,
    from_safetensors: bool = True,
) -> Tuple[Union[AutoModelForCausalLM, xLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Convenience function to load a model, converting it to xLoRA if specified.

    Args:
        model_name (`str`):
            AutoModelForCausalLM pretrained model name or path
        xlora_path (`str`, *optional*):
            Directory to load the xLoRAClassifier from.
        device (`str`):
            Device to load the base model and the xLoRA model to.
        dtype (`torch.dtype`):
            Datatype for the base model.
        use_flash_attention_2 (`bool`, *optional*, defaults to False):
            Use FlashAttention v2 for the base model.
        load_xlora (`bool`, *optional*, defaults to True):
            Load the model to xLoRA.
        verbose (`bool`, *optional*, defaults to False):
            Enable verbose loading.
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
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device,
        verbose=verbose,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_xlora:
        assert xlora_path is not None
        model = from_pretrained(
            load_directory=xlora_path,
            from_safetensors=from_safetensors,
            model=model,
            verbose=verbose,
            device=device,
        )
        if verbose:
            print("X-LoRA loaded.")

    elif verbose:
        print("No X-LoRA loaded, just base model.")

    return model, tokenizer


def load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]:
    """
    Load the scalings log, with awareness to the two types.

    Args:
        path (`str`):
            The path provided to `flush_log_scalings`.
        verbose (`bool`, defaults to `False`)
            Display tqdm.
    """

    output: List[torch.Tensor] = []
    if pathlib.Path(f"{path}-mapping.json").exists():
        with open(f"{path}-mapping.json", "r") as f:
            mapping: Dict[str, List[int]] = json.loads(f.read())

        mapping_full: Dict[int, torch.Tensor] = {}
        maxindex = -1

        if verbose:
            iterator = iter(tqdm.tqdm(mapping.items()))
        else:
            iterator = iter(mapping.items())

        for file, indices in iterator:
            npy_arr = numpy.load(file)
            torch_arr = torch.from_numpy(npy_arr)
            tensors = torch_arr.split(1, dim=0)
            for tensor, index in zip(tensors, indices):
                mapping_full[index] = tensor
                if index > maxindex:
                    maxindex = index

        for index in range(maxindex + 1):
            output.append(mapping_full[index])

    else:
        npy_arr = numpy.load(path)
        torch_arr = torch.from_numpy(npy_arr)
        output.extend(torch_arr.split(1, dim=0))

    return output
