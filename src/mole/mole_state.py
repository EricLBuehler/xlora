from typing import Optional

from torch import Tensor
from typing_extensions import override

from .mole_classifier import MoLEClassifier


class _MoLEScalings:
    def __init__(self, value: Tensor) -> None:
        self.value = value

    @property
    def value(self) -> Tensor:
        return self.value


class _MoLEScalingsWithLifetime(_MoLEScalings):
    def __init__(self, value: Tensor, old_scalings: Tensor, n_accesses_lifetime: int) -> None:
        super().__init__(value)
        self.old_scalings = old_scalings
        self.n_accesses_lifetime = n_accesses_lifetime
        self.n_accesses = 0

    @override
    @property
    def value(self) -> Tensor:
        self.n_accesses += 1
        result = super().value
        if self.n_accesses >= self.n_accesses_lifetime:
            self.value = self.old_value
        return result


_scalings: Optional[_MoLEScalings] = None


def get_scalings() -> Tensor:
    """
    Reads the scaling states (a tensor with 2 dims), raising AssertionError if the state has not been set.
    """
    assert _scalings is not None
    return _scalings.value


def set_scalings(value: Tensor) -> None:
    global _scalings
    """
    Sets the scaling states to a Tensor.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    assert value.ndim == 2
    _scalings = _MoLEScalings(value)


def set_scalings_lifetime(value: Tensor, n_accesses_lifetime: int) -> None:
    global _scalings
    """
    Sets the scaling states to a Tensor. The scaling states will have a lifetime of n accesses.

    A tensor with 2 dim is expected: (batch_size, num_classes)
    """
    assert value.ndim == 2
    _scalings = _MoLEScalingsWithLifetime(value, _scalings.value, n_accesses_lifetime)


_mole_classifier: Optional[MoLEClassifier] = None


def get_mole_classifier() -> MoLEClassifier:
    global _mole_classifier
    """
    Reads the MoLEClassifier.
    """
    assert _mole_classifier is not None
    return _mole_classifier


def set_mole_classifier(value: MoLEClassifier) -> None:
    global _mole_classifier
    """
    Sets the MoLEClassifier.
    """
    _mole_classifier = value
