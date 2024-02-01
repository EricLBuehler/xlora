# X-LoRA
Mixture of LoRA Experts: Leverage the power of fine-tuned LoRA experts by employing a mixture of experts, or MoE technique.

X-LoRA works by learning the alpha scaling values for LoRA adapters, which are frozen. These learned alpha values are used to
gate the LoRA experts in a dense fashion.

X-LoRA is easily applied to any HuggingFace Transformers model.

## Advantages and features
- Effective: Dense gating of experts allows effective mixing
- Efficient fine-tuning: low trainable parameter count
- Hierarchical encapsulated strategy: Re-use existing trained models or model section and re-use them to address complex tasks that cut across experts, following a bio-inspired strategy 
- Easy-to-use API: `add_xlora_to_model`, broad compatibility 

See the [examples](examples) folder for some examples of how to get started with X-LoRA.

## Example
Excerpt from [this](./examples/simple.py) example.
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    device_map="auto",
)

### Convert the model to X-LoRA
model_created = xlora.add_xlora_to_model(
    model=model,
    xlora_config=xlora.xLoRAConfig(config.hidden_size, xlora_depth=8, device=torch.device("cuda")),
    verbose=True,
    adapters={
        "adapter_1": "./path/to/the/checkpoint_adapter_1/",
        "adapter_2": "./path/to/the/checkpoint_adapter_2/",
        "adapter_n": "./path/to/the/checkpoint_adapter_3/",
    },
)
```

### API
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
- `xlora.load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]`
  - Load the scalings log, with awareness to the two types.
- `PeftModel.get_use_trainable_adapters(self) -> bool`
  - Get the trainable or not trainable state of the adapters.

## Installation
Pending a pip release, `git clone` this repository and run `pip install -e .`.
