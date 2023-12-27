from typing import Optional

from torch import Tensor

from utils.mole_utils import is_scalar_tensor


_scaling: Optional[Tensor] = None


@property
def scaling() -> Tensor:
    """
    Reads the scaling state (a tensor with 1 dim), raising AssertionError if the state has not been set.
    """
    assert _scaling is not None
    return _scaling


@scaling.setter
def scaling(value: Tensor) -> None:
    """
    Sets the scaling state to a Tensor. A tensor with 1 dim is expected.
    """
    assert value.ndim == 1
    _scaling = value
