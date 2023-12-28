from typing import Optional

from torch import Tensor


_scalings: Optional[Tensor] = None


def get_scalings() -> Tensor:
    """
    Reads the scaling states (a tensor with 2 dims), raising AssertionError if the state has not been set.
    """
    assert _scalings is not None
    return _scalings


def set_scalings(value: Tensor) -> None:
    """
    Sets the scaling states to a Tensor. A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    assert value.ndim == 2
    _scalings = value
