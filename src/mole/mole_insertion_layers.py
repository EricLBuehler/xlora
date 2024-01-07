import math
from typing import Any, Callable, Dict, List, Optional, Union

import peft
import torch
import torch.nn as nn
from peft.tuners import lora
from peft.tuners.tuners_utils import PeftConfig  # type: ignore
from torch import Tensor

from mole import mole_state

MOLE_ADAPTER_NAME = "mole_adapter"


class MoLEBaseLayer:
    """
    This is a utility class which manages merging LoRA adapters using specified weights.
    """

    @classmethod
    def add_weighted_adapter(
        cls,
        target: lora.LoraLayer,
        adapters: List[str],
        weights: List[Tensor],
        adapter_name: str,
        peft_config: Dict[str, PeftConfig],
        combination_type: str = "svd",
        svd_rank: Optional[bool] = None,
        svd_clamp: Optional[float] = None,
        svd_full_matrices: Optional[bool] = True,
        svd_driver: Optional[str] = None,
    ) -> None:
        for adapter in adapters:
            if adapter not in list(peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks = [peft_config[adapter].r for adapter in adapters]  # type: ignore
        if combination_type == "linear":
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError("All adapters must have the same r value when using `linear` combination_type")
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type == "svd":
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        if adapter_name in target.lora_A:
            target_lora_A = target.lora_A[adapter_name].weight
            target_lora_B = target.lora_B[adapter_name].weight
        elif adapter_name in target.lora_embedding_A:
            target_lora_A = target.lora_embedding_A[adapter_name]
            target_lora_B = target.lora_embedding_B[adapter_name]
        else:
            return

        # TODO(EricLBuehler): Is this correct?
        target_lora_A.data = target_lora_A.data * 0.0  # type: ignore
        # TODO(EricLBuehler): Is this correct?
        target_lora_B.data = target_lora_B.data * 0.0  # type: ignore
        if combination_type == "linear":
            for adapter, weight in zip(adapters, weights):
                if adapter in target.lora_A:
                    current_adapter_lora_A = target.lora_A[adapter].weight
                    current_adapter_lora_B = target.lora_B[adapter].weight
                elif adapter in target.lora_embedding_A:
                    current_adapter_lora_A = target.lora_embedding_A[adapter]
                    current_adapter_lora_B = target.lora_embedding_B[adapter]
                else:
                    continue
                # TODO(EricLBuehler): Is this correct?
                target_lora_A.data += current_adapter_lora_A.data * math.sqrt(weight) * target.scaling[adapter]  # type: ignore
                # TODO(EricLBuehler): Is this correct?
                target_lora_B.data += current_adapter_lora_B.data * math.sqrt(weight)  # type: ignore
        elif combination_type == "cat":
            loras_A: List[Tensor] = []
            loras_B: List[Tensor] = []
            for adapter, weight in zip(adapters, weights):
                if adapter in target.lora_A:
                    current_adapter_lora_A = target.lora_A[adapter].weight
                    current_adapter_lora_B = target.lora_B[adapter].weight
                elif adapter in target.lora_embedding_A:
                    current_adapter_lora_A = target.lora_embedding_A[adapter]
                    current_adapter_lora_B = target.lora_embedding_B[adapter]
                else:
                    continue
                # TODO(EricLBuehler): Is this correct?
                loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])  # type: ignore
                # TODO(EricLBuehler): Is this correct?
                loras_B.append(current_adapter_lora_B.data)  # type: ignore

            if len(loras_A) == 0:
                raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
            loras_A_cat: Tensor = torch.cat(loras_A, dim=0)
            loras_B_cat: Tensor = torch.cat(loras_B, dim=1)
            target_lora_A.data[: loras_A_cat.shape[0], :] = loras_A
            target_lora_B.data[:, : loras_B_cat.shape[1]] = loras_B
        elif combination_type == "svd":
            (
                target_lora_A.data,
                target_lora_B.data,
            ) = cls.svd_weighted_adapter(
                adapters,
                weights,
                new_rank,
                target,
                target_lora_A,
                target_lora_B,
                svd_clamp,
                full_matrices=svd_full_matrices,
                driver=svd_driver,
            )

    @classmethod
    def svd_weighted_adapter(
        cls,
        adapters: List[str],
        weights: List[Tensor],
        new_rank: int,
        target: lora.LoraLayer,
        target_lora_A: Union[Tensor, nn.Module],
        target_lora_B: Union[Tensor, nn.Module],
        clamp: Optional[float] = None,
        full_matrices: Optional[bool] = True,
        driver: Optional[str] = None,
    ):
        valid_adapters = []
        valid_weights = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight)

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")

        delta_weight = valid_weights[0] * target.get_delta_weight(valid_adapters[0])  # type: ignore[attr-defined]
        for adapter, weight in zip(valid_adapters[1:], valid_weights[1:]):
            delta_weight += weight * target.get_delta_weight(adapter)  # type: ignore[attr-defined]
        conv2d = isinstance(target, peft.tuners.lora.layer.Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out:  # type: ignore[attr-defined]
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U


class MoLELayer(MoLEBaseLayer):
    """
    A MoLELayer wraps any LoraLayer and performs the MoLE operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings from mole_state to execute the
    MoLE algorithm. To avoid a RuntimeException, set the scaling state.
    """

    def __init__(
        self,
        adapters: List[str],
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        peft_config: Dict[str, PeftConfig],
        combination_type: str = "svd",
        svd_rank: Optional[bool] = None,
        svd_clamp: Optional[float] = None,
        svd_full_matrices: Optional[bool] = True,
        svd_driver: Optional[str] = None,
        top_k_lora: Optional[int] = None,
    ) -> None:
        self.adapters = adapters
        self.target_forward = target_forward
        self.target = target
        self.peft_config = peft_config
        self.top_k_lora = top_k_lora

        self.combination_type = combination_type
        self.svd_rank = svd_rank
        self.svd_clamp = svd_clamp
        self.svd_full_matrices = svd_full_matrices
        self.svd_driver = svd_driver

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the peft LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the MoLELayer class).
        """
        scalings = mole_state.get_scalings()

        if self.top_k_lora is None:
            for batch_scalings in scalings:
                self.add_weighted_adapter(
                    target=self.target,
                    adapters=self.adapters,
                    weights=list(batch_scalings),
                    adapter_name=MOLE_ADAPTER_NAME,
                    peft_config=self.peft_config,
                    combination_type=self.combination_type,
                    svd_rank=self.svd_rank,
                    svd_clamp=self.svd_clamp,
                    svd_full_matrices=self.svd_full_matrices,
                    svd_driver=self.svd_driver,
                )
        else:
            for batch_scalings in scalings:
                (topk_scalings, indices) = torch.topk(input=batch_scalings, k=self.top_k_lora)
                indices = list(indices)
                adapters = [self.adapters[i] for i in indices]
                self.add_weighted_adapter(
                    target=self.target,
                    adapters=adapters,
                    weights=list(topk_scalings),
                    adapter_name=MOLE_ADAPTER_NAME,
                    peft_config=self.peft_config,
                    combination_type=self.combination_type,
                    svd_rank=self.svd_rank,
                    svd_clamp=self.svd_clamp,
                    svd_full_matrices=self.svd_full_matrices,
                    svd_driver=self.svd_driver,
                )

        self.target.set_adapter(MOLE_ADAPTER_NAME)

        output = self.target_forward(x, *args, **kwargs)

        return output


class BaseTunerWrapper:
    def __init__(self, base_model: peft.tuners.mixed.MixedModel):
        self.model = base_model.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
