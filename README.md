# MoLE
Mixture of LoRA Experts: leverage the power of fine-tuned LoRA experts by employing a mixture of experts, or MoE technique.

MoLE works by learning the alpha scaling values for LoRA adapters, which are frozen. These learned alpha values are used to
gate the LoRA experts in a dense fashion. Optionally, the top-k LoRA experts are selected in a sparse fashion.

## Advantages and features
- Dense gating of experts allows mixing
- Because the MoLE layer is the only trainable layer, fine-tuning has few trainable parameters
- Easy-to-use API: `add_mole_to_model`

See the [examples](examples) folder for some examples of how to get started with MoLE.

## Installation
Pending a pip release, `git clone` this repository and run `pip install -e .`.
