import torch
import xlora
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

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
        "adapter_1": "./path/to/the/checkpoint/",
        "adapter_2": "./path/to/the/checkpoint/",
        "adapter_n": "./path/to/the/checkpoint/",
    },
)

### Set the adapters to trainable
### This means that when loading the model again we should specify only the adapter names.
model_created.set_use_trainable_adapters(True)
model_created.get_use_trainable_adapters()

### Set the scaling pass value
model_created.set_scaling_pass_value(0)

### Reset the scaling pass value
model_created.set_scaling_pass_value(None)

### Example of scalings logging
model_created.enable_scalings_logging()

# Run forward passes to accumulate a log

model_created.flush_log_scalings("./path/to/output/file")

model_created.disable_scalings_logging()

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
