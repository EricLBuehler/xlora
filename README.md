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

## Examples
Excerpt from [this](./examples/simple.ipynb) example.

### Converting a model
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
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

### Loading a trained X-LoRA model *trained without trainable adapters* from scratch
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="auto",
)

model_created = xlora.from_pretrained(
    "./path/to/saved/model",
    model,
    {
        "adapter_1": "./path/to/the/checkpoint/",
        "adapter_2": "./path/to/the/checkpoint/",
        "adapter_n": "./path/to/the/checkpoint/",
    },
    "cuda",
)
```

### Loading a trained X-LoRA model *trained with trainable adapters* from scratch
```python
import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM # type: ignore

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    trust_remote_code=True,
    use_flash_attention_2=False,
    device_map="auto",
)

model_created = xlora.from_pretrained(
    "./path/to/saved/model",
    model,
    ["adapter_1", "adapter_2", "adapter_n"],
    "cuda",
)
```

### Loading a trained X-LoRA model with a convenience function
```python
import torch
from xlora.xlora_utils import load_model  # type: ignore

fine_tune_model_name = "Mistral_v204-rerun_V51Zephyr/checkpoint-420/"

model_loaded, tokenizer = load_model(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    device="cuda:0",
    dtype=torch.bfloat16,
    fine_tune_model_name=fine_tune_model_name,
    adapters={
        "adapter_1": "./path/to/the/checkpoint/",
        "adapter_2": "./path/to/the/checkpoint/",
        "adapter_n": "./path/to/the/checkpoint/",
    },
)
```

### Scalings logging
```python
model: xLoRAModel = ... # Load the model

# Enable scalings logging and begin a log
model.enable_scalings_logging()

# Run forward passes to accumulate a log

# Write the log to a file, or multiple.
model.flush_log_scalings("./path/to/output/file")

# Get a shallow copy of the scalings
log_copy = model.get_scalings_log()

# Disable scalings logging and clear the log
model.disable_scalings_logging()

# Get the latest scalings prediction
scalings_pred = model.get_latest_scalings()
```

### Trainable parameters
```python
model: xLoRAModel = ... # Load the model

num_trainable, num_all_params = model.get_nb_trainable_parameters()

model.print_trainable_parameters()
```

### Setting trainability of adapters dynamically
```python
model: xLoRAModel = ... # Load the model

# Use trainable adapters: mark all adapters as trainable
model.set_use_trainable_adapters(True)

# Get the current status of the trainable adapters, in this case returning True
model.get_use_trainable_adapters()
```

### Setting and resetting the scaling pass value
```python
model: xLoRAModel = ... # Load the model

# Set the scaling pass value to 0, meaning that no adapters will contribute to the scaling pass output
model.set_scaling_pass_value(0)

# Allow the model to use the default scaling pass value
model.set_scaling_pass_value(None)
```

### API
The X-LoRA API is composed of 2 parts: the "Global API" and the "Model API". Generally the global API is used to create X-LoRA models and the model API is used to interface with the models.

## Global API
- `xlora.add_xlora_to_model(model: PreTrainedModel, xlora_config: xLoRAConfig, adapters: Dict[str, str], verbose: bool) -> xLoraModel`
  - Convert a model to an xLoraModel, instantiating the classifier and adapters.
- `xlora.from_pretrained(load_directory: str, model: PreTrainedModel, adapters: Union[List[str], Dict[str, str]], verbose: bool, device: str, from_safetensors: bool = True) -> xLoraModel`
  - Load the X-LoRA classifier and potentially adapters. This should be called after an X-LoRA classifier has been trained.
- `xlora.load_scalings_log(path: str, verbose: bool = False) -> List[torch.Tensor]`
  - Load the scalings log, with awareness to the two types.
- `xlora.xlora_utils.load_model(model_name: str, fine_tune_model_name: str, device: str, dtype: torch.dtype, adapters: Dict[str, str], use_flash_attention_2: bool = False, load_xlora: bool = False, verbose: bool = False, use_cache: bool = False) -> Tuple[Union[AutoModelForCausalLM, xLoRAModel], Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`
  - Convenience function to load a model, converting it to xLoRA if specified.

## Model API
- `xLoraModel.disable_scalings_logging()`
  - Disable scalings logging, clearing the log.
- `xLoraModel.enable_scalings_logging()`
  - Enable scalings logging. Each time a forward pass occurs, the predicted scalings will be logged.
- `xLoraModel.flush_log_scalings(path: str)`
  - Write the scalings log (a tensor of shape (num_logged, batch_size, seq_len, n_layers, n_classes)) to the specified path.
    If the tensor cannot be constructed, multiple files are written containing tensors of shape
    (num_logged, batch_size, seq_len, n_layers, n_classes) such that each file contains one sequence length. Additionally a JSON
    file is outputted containing the mapping from each sequence log file to the index of the contained tensor so that one may reconstruct
    the log order.
    The file specified should not contain an extension.
- `xLoraModel.get_nb_trainable_parameters() -> Tuple[int, int]`
  - Return a tuple `(num_trainable, num_all_params)`
- `xLoraModel.print_scalings_predictions(n_predictions_lifetime: int)`
  - Print the scalings predictions for the next n forward passes of the model.
- `xLoraModel.print_trainable_parameters()`
  - Print the trainable and non-trainable parameters for the given model, including with the X-LoRA components.
- `xLoraModel.set_use_trainable_adapters(use_trainable_adapters: bool)`
  - Set the trainability of the adapters.
- `xLoraModel.set_scaling_pass_value(self, value: Union[Number, None])`
  - Manually set the scalings to a specific value during the scaling pass, forever. Call this function with None to enable the default scalings.
- `xLoraModel.get_use_trainable_adapters(self) -> bool`
  - Get the trainable or not trainable state of the adapters.
- `xLoraModel.get_scalings_log(self) -> List[Tensor]`
  - Returns a shallow copy of the list containing the scalings log. Editing the list does not change the underlying log.
- `xLoraModel.get_latest_scalings(self) -> Optional[Tensor]`
  - Returns the latest scalings prediction, or None if no scalings have been predicted.

## Installation
Pending a pip release, `git clone` this repository and run `pip install -e .`.
