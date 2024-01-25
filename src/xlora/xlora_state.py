from typing import Optional

from torch import Tensor
from typing_extensions import override

from .xlora_classifier import xLoRAClassifier


class _xLoRAScalings:
    def __init__(self, inner: Tensor) -> None:
        self.inner = inner

    @property
    def value(self) -> Tensor:
        return self.inner

    def inc_forward(self):
        ...

    def is_alive(self) -> bool:
        return False


class _xLoRAScalingsWithLifetime(_xLoRAScalings):
    def __init__(self, inner: Tensor, n_accesses_lifetime: int) -> None:
        super().__init__(inner)
        self.n_accesses_lifetime = n_accesses_lifetime
        self.n_accesses = 0

    @override
    def inc_forward(self):
        self.n_accesses += 1

    @override
    def is_alive(self) -> bool:
        return self.n_accesses < self.n_accesses_lifetime


_scalings: Optional[_xLoRAScalings] = None


def get_scalings() -> Tensor:
    """
    Reads the scaling states (a tensor with 2 dims), raising AssertionError if the state has not been set.
    """
    assert _scalings is not None
    return _scalings.value


def inc_forward_scalings():
    assert _scalings is not None
    _scalings.inc_forward()


def set_scalings(value: Tensor) -> None:
    global _scalings
    """
    Sets the scaling states to a Tensor. If the scalings with a lifetime are still alive then the scalings will not be overwritten.

    A tensor with 3 dim is expected: (batch_size, num_layers, num_classes)
    """
    assert value.ndim == 3
    if _scalings is None or not _scalings.is_alive():
        _scalings = _xLoRAScalings(value)


def set_scalings_lifetime(value: Tensor, n_accesses_lifetime: int) -> None:
    global _scalings
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n forward passes.

    A tensor with 3 dim is expected: (batch_size, num_layers, num_classes)
    """
    assert value.ndim == 3
    _scalings = _xLoRAScalingsWithLifetime(value, _scalings.value, n_accesses_lifetime)  # type: ignore


_xlora_classifier: Optional[xLoRAClassifier] = None


def get_xlora_classifier() -> xLoRAClassifier:
    global _xlora_classifier
    """
    Reads the xLoRAClassifier.
    """
    assert _xlora_classifier is not None
    return _xlora_classifier


def set_xlora_classifier(value: xLoRAClassifier) -> None:
    global _xlora_classifier
    """
    Sets the xLoRAClassifier.
    """
    _xlora_classifier = value
