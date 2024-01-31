# X-LoRA
Mixture of LoRA Experts: leverage the power of fine-tuned LoRA experts by employing a mixture of experts, or MoE technique.

X-LoRA works by learning the alpha scaling values for LoRA adapters, which are frozen. These learned alpha values are used to
gate the LoRA experts in a dense fashion.

X-LoRA is easily applied to any HuggingFace Transformers model.

## Advantages and features
- Effective: Dense gating of experts allows effective mixing
- Efficient fine-tuning: low trainable parameter count
- Easy-to-use API: `add_xlora_to_model`

See the [examples](examples) folder for some examples of how to get started with xLoRA.

## API
- `xlora.add_xlora_to_model(model: PreTrainedModel, xlora_config: xLoRAConfig, adapters: Dict[str, str], verbose: bool) -> PeftModel`
  - Convert a model to an xLoRA-model, instantiating the classifier and adapters.
- `PeftModel.disable_scalings_logging()`
  - Disable scalings logging, clearing the log.
- `PeftModel.enable_scalings_logging()`
  - Enable scalings logging. Each time a forward pass occurs, the predicted scalings will be logged.
- `PeftModel.flush_log_scalings(path: str)`
  - Write the scalings log (a tensor of shape (num_logged, batch_size, seq_len, n_layers, n_classes)) to the specified path.
    If the tensor cannot be constructed, multiple files are written containing tensors of shape
    (num_logged, batch_size, seq_len, n_layers, n_classes) such that each file contains one sequence length. Additionally a JSON
    file is outputted containing the mapping from each sequence log file to the index of the contained tensor so that one may reconstruct
    the log order.
    The file specified should not contain an extension.
- `xlora.from_pretrained(load_directory: str, model: PreTrainedModel, adapters: Union[List[str], Dict[str, str]], verbose: bool, device: str, from_safetensors: bool = True) -> PeftModel`
  - Load the xLoRA classifier and potentially adapters. This should be called after an xLoRA classifier has been trained.
- `PeftModel.get_nb_trainable_parameters() -> Tuple[int, int]`
  - Return a tuple `(num_trainable, num_all_params)`
- `PeftModel.print_scalings_predictions(n_predictions_lifetime: int)`
  - Print the scalings predictions for the next n forward passes of the model.
- `PeftModel.print_trainable_parameters()`
  - Print the trainable and non-trainable parameters for the given model, including with the xLoRA components.
- `PeftModel.set_use_trainable_adapters(use_trainable_adapters: bool)`
  - Set the trainability of the adapters.
- `PeftModel.set_scaling_pass_value(self, value: Union[Number, None])`
  - Manually set the scalings to a specific value during the scaling pass, forever. Call this function with None to enable the default  scalings.

### Scalings Logging
```python
# Load model...

# Load xlora...

model.enable_scalings_logging()

# Forward passes...

model.flush_log_scalings(path)
```

### Set trainability of adapters
```python
# Load model... 

# Load xlora...

trainability = ...
model.set_use_trainable_adapters(trainability)
```

## Installation
Pending a pip release, `git clone` this repository and run `pip install -e .`.
