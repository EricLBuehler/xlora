from torch import Tensor


def is_scalar_tensor(tensor: Tensor) -> bool:
    return tensor.ndim == 0
