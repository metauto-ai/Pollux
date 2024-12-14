import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import math
from typing import Tuple
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import inspect

@dataclass
class SchedulerArgs:
    """
    from https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/scheduler/scheduler_config.json
    Note SD3's use_dynamic_shifting is set as False. 
    For Flux, use_dynamic_shifting is set as True.
    """
    num_train_timesteps: int = 1000
    base_image_seq_len: int = 256
    base_shift: float =  0.5
    max_image_seq_len: int = 4096
    max_shift: float = 1.15
    shift: float =  3.0
    weighting_scheme: str = 'logit_normal'
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29
    use_dynamic_shifting: bool = False


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps. 
        from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class RectFlow(torch.nn.Module):
    
    def __init__(self, args: SchedulerArgs):
        super().__init__()
        self.scheduler = self.create_schedulers(args)
        self.weighting_scheme = args.weighting_scheme
        self.logit_mean = args.logit_mean
        self.logit_std = args.logit_std
        self.mode_scale = args.mode_scale
    def create_schedulers(self, args: SchedulerArgs):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps = args.num_train_timesteps,
            base_image_seq_len =  args.base_image_seq_len,
            base_shift =  args.base_shift,
            max_image_seq_len = args.max_image_seq_len,
            max_shift = args.max_shift,
            shift =  args.shift,
            use_dynamic_shifting = args.use_dynamic_shifting
        )
        return scheduler

    def compute_density_for_timestep_sampling(self, batch_size: int)-> torch.Tensor:
        """
        Compute the density for sampling the timesteps when doing SD3 training.

        Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

        SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
        """
        if self.weighting_scheme == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(batch_size,), device="cpu")
            u = torch.nn.functional.sigmoid(u)
        elif self.weighting_scheme == "mode":
            u = torch.rand(size=(batch_size,), device="cpu")
            u = 1 - u - self.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        else:
            u = torch.rand(size=(batch_size,), device="cpu")
        return u
    
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    
    def sample_noised_input(self,x:torch.Tensor)->Tuple[torch.tensor,torch.tensor,torch.tensor]:
        bsz = x.size(0)
        noise = torch.randn_like(x)
        u = self.compute_density_for_timestep_sampling(
            batch_size=bsz,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=x.device)
        sigmas = self.get_sigmas(timesteps, n_dim=x.ndim, dtype=x.dtype)
        noisy_model_input = (1.0 - sigmas) * x + sigmas * noise
        target = noise - x
        return noisy_model_input, timesteps, target