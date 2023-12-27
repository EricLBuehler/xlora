from typing import Optional

from torch import Tensor

from utils.mole_utils import is_scalar_tensor


_scalings: Optional[Tensor] = None


@property
def scalings() -> Tensor:
    """
    Reads the scaling states (a tensor with 1 dim), raising AssertionError if the state has not been set.
    """
    assert _scalings is not None
    return _scalings


@scalings.setter
def scalings(value: Tensor) -> None:
    """
    Sets the scaling states to a Tensor. A tensor with 1 dim is expected.
    """
    assert value.ndim == 1
    _scaling = value
