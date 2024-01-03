# MoLE Documentation: Saving and Loading

## `MoLEModel.save_pretrained`
```python
def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: Optional[bool] = True,
        is_main_process: bool = True,
    ) -> None:
```

This function saves the classifier weights to a directory. It is the counerpart to `load_pretrained`.

```
Args:
    save_directory (`str`):
        Directory where the adapter model and configuration files will be saved (will be created if it does not
        exist).
    safe_serialization (`bool`, *optional*):
        Whether to save the adapter files in safetensors format, defaults to `True`.
    is_main_process (`bool`, *optional*):
        Whether the process calling this is the main process or not. Will default to `True`. Will not save the
        checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
```

The MoLEModel is responsible for saving. Because only the classifier will be trained, only the classifier weights are saved.

## `add_mole_to_model`
```python
def add_mole_to_model(
    model: PreTrainedModel,
    mole_config: MoLEConfig,
    adapters: Dict[str, str],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
) -> MoLEModel:  
```
This method converts all LoRA adapters to MoLE layers, and it is one of the intended entrypoints
for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
errors.

```
Args:
    model (`PreTrainedModel`):
        The model to add the LoRA adapters to. It may be modified in place.
    adapters (`dict`):
        Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
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
Returns:
    model (`MoLEModel`):
        The new model.
```

## `from_pretrained`
```python
def from_pretrained(
    load_directory: str,
    from_safetensors: bool,
    model: PreTrainedModel,
    mole_config: MoLEConfig,
    adapters: Dict[str, str],
    combination_type: str = "svd",
    svd_rank: Optional[bool] = None,
    svd_clamp: Optional[float] = None,
    svd_full_matrices: Optional[bool] = True,
    svd_driver: Optional[str] = None,
) -> MoLEModel:
```

`from_pretrained` initializes the base model like `add_mole_to_model`, and then loads the classifier from specified weights files.

Loads a pretrained classifier from the specified folder while initializing the model. This is the counterpart to [`MoLEModel.save_pretrained`](MOLE_MODEL.md#molemodelsave_pretrained).

This method is very similar to `add_mole_to_model`: it converts all LoRA adapters to MoLE layers, and it is one of
the intended entrypoints for use of MoLE. All LoRA adapters will be frozen, and the MoLEClassifier is initialized.

When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
errors.

```
Args:
    load_directory (`str`):
        The directory to load the classifier weights from.
    from_safetensors (`bool`):
        Whether to load the classifier weights from a .pt or .safetensors file.
    model (`PreTrainedModel`):
        The model to add the LoRA adapters to. It may be modified in place.
    adapters (`dict`):
        Mapping of adapter names to the LoRA adapter id, as per PeftModel.load_adapter. *They will be automatically loaded*, to use as LoRA experts.
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
Returns:
    model (`MoLEModel`):
        The new model.
```
        