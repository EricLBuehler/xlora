# MoLE Documentation: Utility functions

## `get_nb_trainable_parameters`
```python
def get_nb_trainable_parameters(model: PeftMixedModel) -> Tuple[int, int]:
```
Returns the number of trainable parameters and number of all parameters in the model.

## `print_trainable_parameters`
```python
def print_trainable_parameters(model: PeftMixedModel):
```
Prints the number of trainable parameters in the model, including of the MoLE classifier.

## `set_scalings_with_lifetime`
```python
def set_scalings_with_lifetime(value: torch.Tensor, n_accesses_lifetime: int):
```
Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses. Following
this, the value of the scalings will be reset to the previous value. If the original value had a lifetime,
only the value which it would have if it were read at assignment-time will be preserved.

A tensor with 2 dim is expected: (batch_size, num_classes)
    