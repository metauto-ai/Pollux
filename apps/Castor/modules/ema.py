import torch
import math
import copy
from dataclasses import dataclass
from collections import OrderedDict
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh
from typing import Optional, List, Tuple

@dataclass
class EMAArgs:
    decay: float = 0.95
    warmup_steps: int = 2000
    update_buffers: bool = False


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, warmup_steps: int = 0, update_buffers: bool = False):
        """
        Initializes EMA with warmup support.

        Args:
        - model (torch.nn.Module): The model to track.
        - decay (float): Target decay rate (e.g., 0.95).
        - warmup_steps (int): Number of steps for warmup (default is 0 for no warmup).
        """
        self.ema_model = copy.deepcopy(model).eval()  # Duplicate the model for EMA
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.global_step = 0  # Starts at step 0
        self.update_buffers = update_buffers
        # Disable gradient computation for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False


    def _compute_effective_decay(self) -> float:
        """
        Compute the effective decay based on warmup steps.
        """
        if self.warmup_steps > 0:
            return self.decay * (1 - math.exp(-self.global_step / self.warmup_steps))
        return self.decay


    @torch.no_grad()
    def step(self, model: torch.nn.Module):
        """
        Updates the EMA model with the current model parameters.

        Args:
        - model (torch.nn.Module): Current model to update EMA from.
        - update_buffers (bool): Whether to update buffers such as BatchNorm stats.

        # https://github.com/pytorch/pytorch/issues/117742 based on this its okay to update ema model without summoning full parameters
        """
        self.global_step += 1
        effective_decay = self._compute_effective_decay()  # Get the current decay rate

        # Update parameters
        params = OrderedDict(model.named_parameters())
        ema_params = OrderedDict(self.ema_model.named_parameters())

        assert set(ema_params.keys()) == set(params.keys())
        
        

        for name in params:
            ema_params[name].mul_(effective_decay).add_(params[name].data, alpha=1 - effective_decay)

        # Update buffers (if needed)
        if self.update_buffers:
            buffers = OrderedDict(model.named_buffers())
            ema_buffers = OrderedDict(self.ema_model.named_buffers())
            assert set(ema_buffers.keys()) == set(buffers.keys())
            for name in buffers:
                if buffers[name].dtype.is_floating_point:
                    ema_buffers[name].mul_(effective_decay).add_(
                        buffers[name].data, alpha=1 - effective_decay
                    )

    def state_dict(self) -> dict:
        """
        Returns the state dictionary for the EMA model.
        """
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        """
        Loads the state dictionary into the EMA model.
        """
        self.ema_model.load_state_dict(state_dict)

    def to(self, device: torch.device):
        """
        Transfers the EMA model to a specified device.
        """
        self.ema_model.to(device)
