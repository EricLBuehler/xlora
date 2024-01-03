# MoLE Documentation: `MoLEModel(nn.Module)` functions

Utility functions for the `MoLEModel` itself, which inherits from `nn.Module`.

## `MoLEModel.save_pretrained`
```python
def save_pretrained(
    self,
    save_directory: str,
    safe_serialization: Optional[bool] = True,
    is_main_process: bool = True,
) -> None:
```
This function saves the classifier weights to a directory. It is the counerpart to [`from_pretrained`](SAVG_LDG.md#from_pretrained).
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

## `MoLEModel.cuda`
```python
def cuda(self, device: Optional[Union[int, torch.device]] = None):
```
Moves all model and MoLE classifier parameters and buffers to the GPU.

## `MoLEModel.ipu`
```python
def ipu(self, device: Optional[Union[int, torch.device]] = None):
```
Moves all model and MoLE classifier parameters and buffers to the IPU.

## `MoLEModel.xpu`
```python
def xpu(self, device: Optional[Union[int, torch.device]] = None):
```
Moves all model and MoLE classifier parameters and buffers to the XPU.

## `MoLEModel.cpu`
```python
def cpu(self):
```
Moves all model and MoLE classifier parameters and buffers to the CPU.

## `MoLEModel.type`
```python
def type(self, dst_type: Union[torch.dtype, str]):
```
Casts all parameters and buffers of the model and MoLE classifier to `dst_type`. Modifies the models in place.

## `MoLEModel.eval`
```python
def eval(self):
```
Sets the model and MoLE classifier in evaluation mode.

## `MoLEModel.train`
```python
def train(self, mode: Optional[bool] = True)
```
Sets the model and MoLE classifier in training mode if `mode=True`, evaluation mode if `mode=False`.

## `MoLEModel.get_nb_trainable_parameters`
```python
def get_nb_trainable_parameters(self) -> Tuple[int, int]:
```
Returns the number of trainable parameters and number of all parameters in the model.

## `MoLEModel.print_trainable_parameters`
```python
def print_trainable_parameters(self):
```
Prints the number of trainable parameters in the model, including of the MoLE classifier.