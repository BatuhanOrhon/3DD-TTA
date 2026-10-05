"""Shared, label-free encoded/noise inputs for paired GSD sampler arms."""
from dataclasses import dataclass
import hashlib
import json
from typing import Iterator, Tuple

import torch

import tta as baseline


_SCHEDULER_OPTIONS = {
    "beta_end": .02,
    "beta_schedule": "linear",
    "beta_start": .0001,
    "clip_sample": False,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
}


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prepared_input_hash(x: torch.Tensor, shape_latent: torch.Tensor,
                         local_latent: torch.Tensor, style: torch.Tensor,
                         noise: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (x, shape_latent, local_latent, style, noise):
        digest.update(_tensor_hash(tensor).encode("ascii"))
    return digest.hexdigest()


def _prepared_component_hashes(x: torch.Tensor, shape_latent: torch.Tensor,
                               local_latent: torch.Tensor, style: torch.Tensor,
                               noise: torch.Tensor, timesteps: torch.Tensor,
                               alpha_bar: torch.Tensor) -> dict:
    return {
        "input_points": _tensor_hash(x),
        "shape_latent": _tensor_hash(shape_latent),
        "local_latent": _tensor_hash(local_latent),
        "style_conditioning": _tensor_hash(style),
        "local_noise": _tensor_hash(noise),
        "timesteps": _tensor_hash(timesteps),
        "alpha_bar": _tensor_hash(alpha_bar),
    }


def _config_hash(total: int, steps_back_local: int, timesteps: torch.Tensor,
                 alpha_bar: torch.Tensor) -> str:
    payload = {
        "schema_version": 1,
        "scheduler": _SCHEDULER_OPTIONS,
        "total": int(total),
        "steps_back_local": int(steps_back_local),
        "timesteps": [int(value) for value in timesteps.detach().cpu().tolist()],
        "alpha_bar": float(alpha_bar.detach().cpu().item()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                         .encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedGuidanceInputs:
    """Detached common inputs; each sampler arm clones these before stepping."""

    shape_latent: torch.Tensor
    local_latent: torch.Tensor
    style_conditioning: torch.Tensor
    local_noise: torch.Tensor
    timesteps: torch.Tensor
    alpha_bar: torch.Tensor
    scheduler_options: Tuple[Tuple[str, object], ...]
    total: int
    steps_back_local: int
    input_sha256: str
    component_sha256: dict
    config_sha256: str

    def tensor_items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for name in ("shape_latent", "local_latent", "style_conditioning",
                     "local_noise", "timesteps", "alpha_bar"):
            yield name, getattr(self, name)

    def clone_states(self):
        return (self.shape_latent.detach().clone(), self.local_latent.detach().clone(),
                self.style_conditioning.detach().clone(), self.local_noise.detach().clone(),
                self.timesteps.detach().clone(), self.alpha_bar.detach().clone())

    def validate_for(self, x: torch.Tensor, total: int, steps_back_local: int) -> None:
        expected_input_hash = _prepared_input_hash(
            x, self.shape_latent, self.local_latent,
            self.style_conditioning, self.local_noise)
        if self.input_sha256 != expected_input_hash:
            raise ValueError("prepared inputs do not match the input point cloud")
        if self.total != total or self.steps_back_local != steps_back_local:
            raise ValueError("prepared inputs do not match the sampler configuration")
        expected = _config_hash(total, steps_back_local, self.timesteps, self.alpha_bar)
        if self.config_sha256 != expected:
            raise ValueError("prepared scheduler/config identity is invalid")
        if self.scheduler_options != tuple(sorted(_SCHEDULER_OPTIONS.items())):
            raise ValueError("prepared scheduler contract is invalid")
        batch = x.shape[0]
        if (self.shape_latent.shape[0] != batch or self.local_latent.shape != (batch, 8192, 1, 1)
                or self.local_noise.shape != self.local_latent.shape
                or self.style_conditioning.shape[0] != batch
                or self.timesteps.numel() != (total * steps_back_local) // 100):
            raise ValueError("prepared input tensor shapes do not match the sampler")
        # Diffusers keeps alphas_cumprod on the scheduler's device, often CPU;
        # the sampler explicitly transfers this scalar to the latent device.
        for name, value in self.tensor_items():
            if name != "alpha_bar" and value.device != x.device:
                raise ValueError("prepared {} and point cloud must share a device".format(name))


@torch.no_grad()
def prepare_guidance_inputs(x: torch.Tensor, lion, *, total: int,
                            steps_back_local: int) -> PreparedGuidanceInputs:
    """Encode once and draw local noise once using the existing DDIM contract."""
    if total <= 0 or not 0 < steps_back_local <= 100:
        raise ValueError("GSD requires total > 0 and steps_back_local in (0, 100]")
    reverse_steps = (total * steps_back_local) // 100
    if reverse_steps < 1:
        raise ValueError("GSD requires at least one reverse step")
    vae, prior = lion.vae, lion.priors[1]
    baseline.grad_freeze(vae)
    baseline.grad_freeze(prior)
    vae.eval()
    prior.eval()
    scheduler = baseline.DDIMScheduler(**_SCHEDULER_OPTIONS)
    scheduler.set_timesteps(total, device=x.device)
    timesteps = scheduler.timesteps[-reverse_steps:].detach().clone()
    alpha_bar = scheduler.alphas_cumprod[timesteps[0]].detach().clone()
    latents = vae.encode(x)
    shape_latent = latents[2][0][0].unsqueeze(2).unsqueeze(3).detach().clone()
    local_latent = latents[2][1][0].unsqueeze(2).unsqueeze(3).detach().clone()
    if local_latent.shape != (len(x), 8192, 1, 1):
        raise ValueError("GSD expects LION local encoding B x 8192 x 1 x 1")
    style = vae.global2style(shape_latent).detach().clone()
    noise = torch.randn_like(local_latent).detach()
    component_hashes = _prepared_component_hashes(
        x, shape_latent, local_latent, style, noise, timesteps, alpha_bar)
    return PreparedGuidanceInputs(
        shape_latent=shape_latent,
        local_latent=local_latent,
        style_conditioning=style,
        local_noise=noise,
        timesteps=timesteps,
        alpha_bar=alpha_bar,
        scheduler_options=tuple(sorted(_SCHEDULER_OPTIONS.items())),
        total=total,
        steps_back_local=steps_back_local,
        input_sha256=_prepared_input_hash(x, shape_latent, local_latent, style, noise),
        component_sha256=component_hashes,
        config_sha256=_config_hash(total, steps_back_local, timesteps, alpha_bar),
    )
