"""
MimicDiffusion Purification (arXiv 2312.04802, Algorithm 1).

Reverse diffusion loop with Manhattan-distance guidance:

    x_{t-1} = mu_theta(x_t, t) + sigma_t^2 * (g_long + g_short)

    g_long  = lambda * sign(x_adv - x_hat_0)   [long-range: global structure]
    g_short = lambda * sign(x_adv - x_t)        [short-range: local artifacts]

Guidance is applied only within the interval [step_s, step_e].
The L1/sign guidance is chosen because sign(x_ori - x_t) ~= sign(x_adv - x_t)
for small perturbations (Lemma 1 in the paper).

Backbone diffusion model: google/ddpm-cifar10-32 (publicly available on HF Hub).
That model was trained on CIFAR-10 but acts as a generic image prior; it still
substantially purifies CIFAR-100 adversarial examples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Cosine noise schedule (same as the DDPM paper, OpenAI variant)
# ---------------------------------------------------------------------------

def make_cosine_schedule(T: int = 1000, s: float = 0.008) -> dict:
    """Return alpha_bar, alpha, beta tensors for a cosine noise schedule."""
    steps     = torch.arange(T + 1, dtype=torch.float64)
    f         = torch.cos(((steps / T) + s) / (1.0 + s) * torch.pi / 2) ** 2
    alpha_bar = (f / f[0]).clamp(min=1e-5).float()
    beta      = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    alpha     = 1.0 - beta
    return dict(
        alpha_bar = alpha_bar[1:],   # length T
        alpha     = alpha,
        beta      = beta,
    )


# ---------------------------------------------------------------------------
# Diffusion helpers
# ---------------------------------------------------------------------------

def q_sample(x0: torch.Tensor, t: int, sch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: sample x_t from x_0 at timestep t."""
    ab  = sch["alpha_bar"][t].to(x0.device)
    eps = torch.randn_like(x0)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps, eps


def predict_x0(
    eps_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    sch: dict,
) -> torch.Tensor:
    """Compute x_hat_0 from the predicted noise eps and x_t."""
    ab = sch["alpha_bar"][t].to(x_t.device)
    return ((x_t - (1.0 - ab).sqrt() * eps_pred) / ab.sqrt()).clamp(-1.0, 1.0)


def ddpm_reverse_mean(
    eps_theta: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    sch: dict,
) -> torch.Tensor:
    """DDPM posterior mean mu_theta(x_t, t) using the predicted noise."""
    ab_t    = sch["alpha_bar"][t].to(x_t.device)
    ab_prev = (
        sch["alpha_bar"][t - 1].to(x_t.device) if t > 0
        else torch.tensor(1.0, device=x_t.device)
    )
    beta_t  = sch["beta"][t].to(x_t.device)
    alpha_t = sch["alpha"][t].to(x_t.device)
    x0      = predict_x0(eps_theta, x_t, t, sch)
    mu      = (
        (ab_prev.sqrt() * beta_t) / (1.0 - ab_t) * x0
        + (alpha_t.sqrt() * (1.0 - ab_prev)) / (1.0 - ab_t) * x_t
    )
    return mu


# ---------------------------------------------------------------------------
# MimicDiffusion purifier
# ---------------------------------------------------------------------------

class MimicDiffusionPurifier:
    """
    Algorithm 1 from arXiv 2312.04802.

    Parameters
    ----------
    eps_model  : callable (x_t, t_tensor) -> eps_pred
                 Wraps the UNet; returns predicted noise for a batch.
    T          : total diffusion timesteps (1000 matches the pretrained model)
    step_s     : guidance window start (default 100)
    step_e     : guidance window end   (default 600)
    lam        : guidance strength lambda
    device     : torch device
    """

    def __init__(
        self,
        eps_model,
        T: int = 1000,
        step_s: int = 100,
        step_e: int = 600,
        lam: float = 0.8,
        device: str | torch.device = "cpu",
    ):
        self.eps_model = eps_model
        self.T         = T
        self.step_s    = step_s
        self.step_e    = step_e
        self.lam       = lam
        self.device    = torch.device(device)
        self.sch       = make_cosine_schedule(T)

    @torch.no_grad()
    def purify(
        self,
        x_adv: torch.Tensor,
        t_start: int = 150,
        progress_callback=None,
    ) -> torch.Tensor:
        """
        Purify a batch of adversarial images.

        x_adv          : (B, 3, 32, 32) normalised adversarial images
        t_start        : how many noisy steps to add before reversing
                         (150 works well for FGSM/PGD)
        progress_callback: optional callable(step, total) for UI progress bars

        Returns purified tensor, same shape as x_adv, clamped to [-1, 1].
        """
        x_adv = x_adv.to(self.device)
        B     = x_adv.shape[0]

        # Add noise up to t_start
        x_t, _ = q_sample(x_adv, t_start, self.sch)
        total   = t_start + 1

        for i, t in enumerate(reversed(range(0, t_start + 1))):
            t_in = torch.full((B,), t, dtype=torch.long, device=self.device)
            eps  = self.eps_model(x_t, t_in)
            mu   = ddpm_reverse_mean(eps, x_t, t, self.sch)

            # Apply Manhattan-distance guidance within [step_s, step_e]
            if self.step_s <= t <= self.step_e:
                x0h = predict_x0(eps, x_t, t, self.sch)
                g_l = self.lam * torch.sign(x_adv - x0h)   # long-range
                g_s = self.lam * torch.sign(x_adv - x_t)   # short-range
                sig = self.sch["beta"][t].to(self.device).sqrt()
                mu  = mu + (sig ** 2) * (g_l + g_s)

            if t > 0:
                sig = self.sch["beta"][t].to(self.device).sqrt()
                x_t = mu + sig * torch.randn_like(x_t)
            else:
                x_t = mu

            if progress_callback is not None:
                progress_callback(i + 1, total)

        return x_t.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Lazy model loader (cached across Streamlit runs via st.cache_resource)
# ---------------------------------------------------------------------------

def load_ddpm_unet(device: torch.device):
    """
    Load google/ddpm-cifar10-32 from HuggingFace Hub.
    Returns a callable eps_model_fn(x_t, t_tensor) -> eps_pred.
    """
    from diffusers import UNet2DModel

    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    def eps_model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return unet(x_t, t).sample

    return eps_model_fn


def build_purifier(device: torch.device, lam: float = 0.8) -> MimicDiffusionPurifier:
    """Convenience factory: download DDPM and wrap in a purifier."""
    eps_fn = load_ddpm_unet(device)
    return MimicDiffusionPurifier(
        eps_model = eps_fn,
        T         = 1000,
        step_s    = 100,
        step_e    = 600,
        lam       = lam,
        device    = device,
    )
