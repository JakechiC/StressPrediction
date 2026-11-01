"""A readout-only PaiNN intervention with a signed, symmetric tensor head."""

import torch
from torch import nn

from .painn import PaiNNDirect
from .painn_utils import ScaledSiLU


class SignedDyadStressOutput(nn.Module):
    r"""Predict p I + sum_k a_k (u_k u_k^T - ||u_k||^2 I / 3), k=1..3.

    The three scalar coefficients have no sign constraint. The vectors are
    bias-free linear combinations of the final PaiNN vector channels. This
    constructs a second-order tensor; it does not constrain matrix rank to two.
    The original isotropic predictor is supplied unchanged by PaiNNHeadOnly.
    """

    def __init__(self, hidden_channels: int, iso_predictor: nn.Linear) -> None:
        super().__init__()
        if hidden_channels < 2:
            raise ValueError("SignedDyadStressOutput needs at least two channels")
        self.iso_predictor = iso_predictor
        self.vector_projection = nn.Linear(hidden_channels, 3, bias=False)
        self.coefficients = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 3),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        vectors = self.vector_projection(vec).transpose(1, 2)  # [N, 3 dyads, 3 xyz]
        dyads = vectors.unsqueeze(-1) * vectors.unsqueeze(-2)
        squared_norms = vectors.square().sum(dim=-1)
        identity = torch.eye(3, device=x.device, dtype=x.dtype)
        traceless_dyads = dyads - squared_norms[..., None, None] * identity / 3.0
        weights = self.coefficients(x)
        deviatoric = (weights[..., None, None] * traceless_dyads).sum(dim=1)
        pressure = self.iso_predictor(x)
        return pressure.unsqueeze(-1) * identity + deviatoric


class PaiNNHeadOnly(PaiNNDirect):
    """Keep PaiNNDirect's entire forward/backbone and replace only its W head.

    Constructing the parent first preserves its exact shared initialization,
    including the energy head's final reset. CPU RNG state is restored after
    constructing the new head. The trainer constructs models on CPU before
    moving them to the allocated device, so no CUDA RNG is consumed here.
    The parent's isotropic predictor is reused with its initial values intact.

    Shared parameters remain trainable: their initial values and architecture
    are paired, but their values can diverge during joint E+W optimization.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_rbf: int = 64,
        cutoff: float = 5.0,
        rbf: dict[str, str] | None = None,
        envelope: dict[str, str | int] | None = None,
        num_elements: int = 86,
    ) -> None:
        super().__init__(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
            num_elements=num_elements,
        )
        isotropic = self.out_stress_feature.iso_predictor
        with torch.random.fork_rng(devices=[]):
            self.out_stress_feature = SignedDyadStressOutput(hidden_channels, isotropic)


__all__ = ["PaiNNHeadOnly", "SignedDyadStressOutput"]
