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
- `xlora.disable_scalings_logging()`
- `xlora.enable_scalings_logging()`
- `xlora.flush_log_scalings(path: str)`
- `xlora.from_pretrained(load_directory: str, model: PreTrainedModel, adapters: Union[List[str], Dict[str, str]], verbose: bool, device: str, from_safetensors: bool = True) -> PeftModel`
- `xlora.get_nb_trainable_parameters(model: PeftModel) -> Tuple[int, int]`
- `xlora.print_scalings_predictions(n_predictions_lifetime: int)`
- `xlora.print_trainable_parameters(model: PeftModel)`
- `xlora.set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int)`
- `PeftModel.set_use_trainable_adapters(use_trainable_adapters: bool)`

## Installation
Pending a pip release, `git clone` this repository and run `pip install -e .`.
