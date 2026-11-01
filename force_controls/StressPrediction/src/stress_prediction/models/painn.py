import torch
from torch import nn
import math
from torch_scatter import scatter
from torch.autograd import grad

from .painn_utils import (
    PaiNNMessage,
    PaiNNUpdate,
    AtomEmbedding,
    RadialBasis,
    ScaledSiLU,
    EquivariantStressOutput,
    EdgeBasedStressOutput,
)

__all__ = [
    "PaiNNDerivative",
    "PaiNNDirect",
    "PaiNNEdge",
]


class PaiNNDirect(nn.Module):
    r"""PaiNN backbone with the original single-dyad direct readout, p I + v v^T.

    This restricted readout is distinct from PaiNNEdge. The default anisotropic
    increment has matrix rank at most one; its two smaller eigenvalues coincide.

    PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
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
        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}
        if rbf is None:
            rbf = {"name": "gaussian"}
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_stress_feature = EquivariantStressOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_scalar_feature[0].weight)
        self.out_scalar_feature[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_scalar_feature[2].weight)
        self.out_scalar_feature[2].bias.data.fill_(0)

    @torch.enable_grad()
    def forward(self, data):

        atomic_numbers = data.z
        positions = data.positions
        edge_index = data.edge_index
        shifts = data.shifts
        num_atoms = data.num_nodes

        src, tgt = edge_index

        # compute atom and pair features
        r_ij = positions[tgt] - positions[src] + shifts

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-7)

        edge_dist = d_ij.squeeze(-1)
        edge_vector = dir_ij

        z = atomic_numbers.long()

        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec

        #### Output block #####################################################

        energy_per_atom = self.out_scalar_feature(x)  # [N, 1]
        stress_per_atom = self.out_stress_feature(x, vec)

        return (
            energy_per_atom,
            stress_per_atom,
        )  # Model-space energy and virial; both have energy dimensions.

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"cutoff={self.cutoff})"
        )


class PaiNNEdge(nn.Module):
    r"""PaiNN with learned invariant coefficients multiplying edge dyads.

    Coefficients use both endpoint scalar features; no extra output envelope is
    applied beyond the radial envelope in message passing.

    PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
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
        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}
        if rbf is None:
            rbf = {"name": "gaussian"}
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_stress_feature = EdgeBasedStressOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_scalar_feature[0].weight)
        self.out_scalar_feature[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_scalar_feature[2].weight)
        self.out_scalar_feature[2].bias.data.fill_(0)

    @torch.enable_grad()
    def forward(self, data):

        atomic_numbers = data.z
        positions = data.positions
        edge_index = data.edge_index
        shifts = data.shifts
        num_atoms = data.num_nodes

        src, tgt = edge_index

        # compute atom and pair features
        r_ij = positions[tgt] - positions[src] + shifts

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-7)

        edge_dist = d_ij.squeeze(-1)
        edge_vector = dir_ij

        z = atomic_numbers.long()

        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec

        #### Output block #####################################################

        energy_per_atom = self.out_scalar_feature(x)  # [N, 1]
        stress_per_atom = self.out_stress_feature(x, edge_index, edge_vector)

        return (
            energy_per_atom,
            stress_per_atom,
        )  # Model-space energy and virial; both have energy dimensions.

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"cutoff={self.cutoff})"
        )


class PaiNNDerivative(nn.Module):
    r"""PaiNN local virials from relative-edge derivatives of the scalar energy.

    The inherited sender/receiver averaging, symmetrization, and sign convention
    are preserved. Evaluation requires autograd, so avoid inference_mode().

    PaiNN model based on the description in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties
    and molecular spectra, https://arxiv.org/abs/2102.03150.
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
        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}
        if rbf is None:
            rbf = {"name": "gaussian"}
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                PaiNNMessage(hidden_channels, num_rbf).jittable()
            )
            self.update_layers.append(PaiNNUpdate(hidden_channels))

        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.out_scalar_feature[0].weight)
        self.out_scalar_feature[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_scalar_feature[2].weight)
        self.out_scalar_feature[2].bias.data.fill_(0)

    @torch.enable_grad()
    def forward(self, data):

        atomic_numbers = data.z
        positions = data.positions
        edge_index = data.edge_index
        shifts = data.shifts
        num_atoms = data.num_nodes

        src, tgt = edge_index

        # compute atom and pair features
        r_ij = positions[tgt] - positions[src] + shifts

        r_ij = r_ij.requires_grad_(True)  # for edge forces

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / (d_ij + 1e-7)

        edge_dist = d_ij.squeeze(-1)
        edge_vector = dir_ij

        z = atomic_numbers.long()

        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec

        #### Output block #####################################################

        energy_per_atom = self.out_scalar_feature(x)  # [N, 1]
        energy = scatter(energy_per_atom, data.batch, 0, dim_size=data.num_graphs)

        # edge forces
        dEdrij = grad(
            energy,
            r_ij,
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training,
        )[0]

        if dEdrij is None:
            edge_forces = torch.zeros_like(r_ij)
        else:
            edge_forces = -1 * dEdrij

        edge_virial = torch.einsum("zi,zj->zij", edge_forces, r_ij)
        atom_virial_sender = scatter(
            src=edge_virial,
            index=edge_index[0],
            dim=0,
            dim_size=num_atoms,
        )
        atom_virial_receiver = scatter(
            src=edge_virial, index=edge_index[1], dim=0, dim_size=num_atoms
        )
        atom_virial = (atom_virial_sender + atom_virial_receiver) / 2
        atom_virial = (atom_virial + atom_virial.transpose(-1, -2)) / 2

        return (
            energy_per_atom,
            -1 * atom_virial,
        )  # Model-space energy and virial; both have energy dimensions.

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"cutoff={self.cutoff})"
        )
