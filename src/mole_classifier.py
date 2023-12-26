import torch.nn as nn
import torch


class MoleClassifier(nn.Module):
    """
    A classifier to select LoRA layers for MoLE
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
