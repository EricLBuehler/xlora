from typing import Optional

from torch import Tensor

from mole.mole.mole_classifier import MoLEClassifier

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


_mole_classifier: Optional[MoLEClassifier] = None


def get_mole_classifier() -> MoLEClassifier:
    """
    Reads the MoLEClassifier.
    """
    assert _mole_classifier is not None
    return _mole_classifier


def set_mole_classifier(value: MoLEClassifier) -> None:
    """
    Sets the MoLEClassifier.
    """
    _mole_classifier = value
