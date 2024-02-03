import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
from xlora.xlora_utils import load_model  # type: ignore

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
        "adapter_1": "./path/to/the/checkpoint/",
        "adapter_2": "./path/to/the/checkpoint/",
        "adapter_n": "./path/to/the/checkpoint/",
    },
)

### Set the adapters to trainable
### This means that when loading the model again we should specify only the adapter names.
# Use trainable adapters: mark all adapters as trainable
model_created.set_use_trainable_adapters(True)

# Get the current status of the trainable adapters, in this case returning True
model_created.get_use_trainable_adapters()

### Set and resetting the scaling pass value
# Set the scaling pass value to 0, meaning that no adapters will contribute to the scaling pass output
model_created.set_scaling_pass_value(0)

# Allow the model to use the default scaling pass value
model_created.set_scaling_pass_value(None)

### Example of scalings logging
# Enable scalings logging and begin a log
model_created.enable_scalings_logging()

# Run forward passes to accumulate a log

# Write the log to a file, or multiple.
model_created.flush_log_scalings("./path/to/output/file")

# Get a shallow copy of the scalings
log_copy = model_created.get_scalings_log()

# Disable scalings logging and clear the log
model_created.disable_scalings_logging()

### Example of getting, printing trainable parameters
num_trainable, num_all_params = model_created.get_nb_trainable_parameters()

model_created.print_trainable_parameters()

### From pretrained for models trained with `model_created.get_use_trainable_adapters() == True`
loaded_model = xlora.from_pretrained("./path/to/saved/model", model, ["adapter_1", "adapter_2"], "cuda")

### From pretrained for models trained with `model_created.get_use_trainable_adapters() == False`
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


### Using the simpler API
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
