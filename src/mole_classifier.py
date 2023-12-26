import torch.nn as nn
import torch


class MoleClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass
