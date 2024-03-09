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

from xlora.xlora_config import xLoRAConfig  # type: ignore

from .xlora import from_pretrained, xLoRAModel  # type: ignore


def _get_file_path_single(
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


def _get_file_path(load_directory: str, name: str, dir: Optional[str]) -> str:
    if dir is not None:
        return _get_file_path_dir(load_directory, name, dir)
    return _get_file_path_single(load_directory, name)


def load_model(
    model_name: str,
    device: str,
    dtype: torch.dtype,
    adapters: Dict[str, str],
    use_flash_attention_2: bool = False,
    load_xlora: bool = True,
    verbose: bool = False,
    from_safetensors: bool = True,
    hf_hub_subdir: Optional[str] = None,
) -> Tuple[Union[AutoModelForCausalLM, xLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    """
    Convenience function to load a model with the specified adapters like the X-LoRA config, converting it to xLoRA if specified.

    Args:
        model_name (`str`):
            Directory or HF model repo ID to load the xLoRA model from.
        device (`str`):
            Device to load the base model and the xLoRA model to.
        dtype (`torch.dtype`):
            Datatype for the base model.
        adapters (`list` or `dict`, defaults to None):
            Specify a mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
            Specify the list if the adapters were trainable. Specify this parameter to override use of the trained adapters.
        use_flash_attention_2 (`bool`, *optional*, defaults to False):
            Use FlashAttention v2 for the base model.
        load_xlora (`bool`, *optional*, defaults to True):
            Load the model to xLoRA.
        verbose (`bool`, *optional*, defaults to False):
            Enable verbose loading.
        from_safetensors (`bool`, *optional*, defaults to True):
            Whether to load the classifier weights from a .pt or .safetensors file.
        hf_hub_subdir (`str`, *optional*, defaults to None):
            If `model_name` is a HF model repo ID, specify a subdirectory where the xLoRA config and classifier may be found.

    Returns:
        Tuple whose elements are respectively:

        model (`AutoModelForCausalLM` or `xLoRAModel`):
            The model.

        tokenizer (`AutoTokenizer`):
            The tokenizer.
    """
    with open(_get_file_path(model_name, "xlora_config.json", hf_hub_subdir), "r") as f:
        conf = json.load(f)
        conf["device"] = torch.device(device)

        if "adapters" not in conf:
            conf["adapters"] = adapters
        xlora_config = xLoRAConfig(**conf)

    model = AutoModelForCausalLM.from_pretrained(
        xlora_config.base_model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=dtype,
        use_flash_attention_2=use_flash_attention_2,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        xlora_config.base_model_id,
        trust_remote_code=True,
        device_map=device,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if load_xlora:
        model = from_pretrained(
            load_directory=model_name,
            from_safetensors=from_safetensors,
            model=model,
            verbose=verbose,
            device=device,
            hf_hub_subdir=hf_hub_subdir,
            adapters=adapters,
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
