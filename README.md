# xLoRA
Mixture of LoRA Experts: leverage the power of fine-tuned LoRA experts by employing a mixture of experts, or MoE technique.

xLoRA works by learning the alpha scaling values for LoRA adapters, which are frozen. These learned alpha values are used to
gate the LoRA experts in a dense fashion. Optionally, the top-k LoRA experts may be selected in a sparse fashion based on the 
output of the xLoRA classifier.

xLoRA is easily applied to any HuggingFace Transformers model.

## Advantages and features
- Effective: Dense gating of experts allows effective mixing
- Efficient fine-tuning: low trainable parameter count.
- Easy-to-use API: `add_xlora_to_model`

See the [examples](examples) folder for some examples of how to get started with xLoRA.

## API
- `xlora.add_xlora_to_model(model: PreTrainedModel, xlora_config: xLoRAConfig, adapters: Dict[str, str], verbose: bool) -> PeftModel`
  - Convert a model to an xLoRA-model, instantiating the classifier and adapters.
- `PeftModel.disable_scalings_logging()`
  - Disable scalings logging, clearing the log.
- `PeftModel.enable_scalings_logging()`
  - Enable scalings logging.
- `PeftModel.flush_log_scalings(path: str)`
  - Save the log scalings as a tensor of `[log_length, batch_size, num_layers, num_scalings]`. Flushes the log.
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

### Scalings Logging
```python
# Load model...

# Load xlora...

xlora.enable_scalings_logging()

# Forward passes...

xlora.flush_log_scalings(path)
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
