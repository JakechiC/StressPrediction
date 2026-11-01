import torch
from torch import nn
import math
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from .painn_utils import AtomEmbedding, RadialBasis, ScaledSiLU

__all__ = [
    "TensorPaiNN",
]


class TensorPaiNNMessage(MessagePassing):
    """Propagate scalar [N,C], vector [N,3,C], and tensor [N,3,3,C] features.

    Five channel groups gate scalar messages, existing vectors, edge vectors,
    existing tensors, and edge dyads, respectively."""

    def __init__(
        self,
        hidden_channels,
        num_rbf,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)

        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 5),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 5)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, tsr, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))  # [N, 5C]
        rbfh = self.rbf_proj(edge_rbf)  # [E, 5C]

        # propagate_type: (xh: Tensor, vec: Tensor, tsr: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec, dtsr = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            tsr=tsr,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )

        return dx, dvec, dtsr

    def message(self, xh_j, vec_j, tsr_j, rbfh_ij, r_ij):
        # xh_j: [E, 5C]  rbfh_ij: [E, 5C]
        x, xh2, xh3, xh4, xh5 = torch.split(
            xh_j * rbfh_ij, self.hidden_channels, dim=-1
        )
        # x, xh2..xh5: [E, C]

        xh2 = xh2 * self.inv_sqrt_3

        # vec_j: [E, 3, C]  r_ij: [E, 3]
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        # [E, 3, C] + [E, 1, C] * [E, 3, 1] -> [E, 3, C]
        vec = vec * self.inv_sqrt_h

        # tsr_j: [E, 3, 3, C]

        outer = r_ij.unsqueeze(2) * r_ij.unsqueeze(
            1
        )  # [E, 3, 1] * [E, 1, 3] -> [E, 3, 3]

        # xh4/xh5: [E, C] -> unsqueeze(1).unsqueeze(1) -> [E, 1, 1, C]
        tsr = tsr_j * xh4.unsqueeze(1).unsqueeze(1) + xh5.unsqueeze(1).unsqueeze(
            1
        ) * outer.unsqueeze(-1)
        # [E, 3, 3, C] + [E, 1, 1, C] * [E, 3, 3, 1] -> [E, 3, 3, C]

        return x, vec, tsr

    def aggregate(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        dim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, vec, tsr = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        tsr = scatter(tsr, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec, tsr

    def update(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return inputs


class TensorPaiNNUpdate(nn.Module):
    """Update scalar, vector, and tensor channels jointly.

    Scalar updates depend on vector and tensor norms and inner products.
    This changes the backbone, not only the final tensor readout."""

    def __init__(self, hidden_channels) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)

        self.tsr_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)

        self.xvectsr_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 5),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.tsr_proj.weight)
        nn.init.xavier_uniform_(self.xvectsr_proj[0].weight)
        self.xvectsr_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvectsr_proj[2].weight)
        self.xvectsr_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, tsr):

        # vec: [N, 3, C]  -> vec_proj: [N, 3, 2C]
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        # vec1, vec2: [N, 3, C]
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # tsr: [N, 3, 3, C]  -> tsr_proj: [N, 3, 3, 2C]
        tsr_1, tsr_2 = torch.split(self.tsr_proj(tsr), self.hidden_channels, dim=-1)
        # tsr_1, tsr_2: [N, 3, 3, C]

        tsr_dot = (tsr_1 * tsr_2).sum(dim=(1, 2)) * self.inv_sqrt_h
        # [N, C]

        vec2_norm = torch.sqrt(torch.sum(vec2**2, dim=1) + 1e-8)  # [N, C]

        tsr2_norm = torch.sqrt(torch.sum(tsr_2**2, dim=(1, 2)) + 1e-8)  # [N, C]

        x_vt_h = self.xvectsr_proj(
            torch.cat([x, vec2_norm, tsr2_norm], dim=-1)  # [N, 3C]
        )  # [N, 5C]
        xvec1, xvec2, xvec3, xtsr4, xtsr5 = torch.split(
            x_vt_h, self.hidden_channels, dim=-1
        )

        dx = xvec1 + xvec2 * vec_dot + xtsr4 * tsr_dot  # [N, C]
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1  # [N, 1, C] * [N, 3, C] -> [N, 3, C]

        dtsr = (
            xtsr5.unsqueeze(1).unsqueeze(1) * tsr_1
        )  # [N, 1, 1, C] * [N, 3, 3, C] -> [N, 3, 3, C]

        return dx, dvec, dtsr


class TensorStressOutput(nn.Module):
    """Read out a symmetric local virial from all three feature families.

    W = alpha(x) I + linear(tsr)
        + channel_sum(v1 outer v2 + v2 outer v1).
    Tensor features are symmetric but are not constrained to be traceless."""

    def __init__(self, hidden_channels: int):
        super().__init__()

        self.iso_proj = nn.Linear(hidden_channels, 1)

        self.tsr_proj = nn.Linear(hidden_channels, 1, bias=False)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)

        self.vec_out = nn.Linear(hidden_channels, 1, bias=False)

    def forward(
        self, x: torch.Tensor, vec: torch.Tensor, tsr: torch.Tensor
    ) -> torch.Tensor:
        # x  : [N, C]
        # vec: [N, 3, C]
        # tsr: [N, 3, 3, C]

        alpha = self.iso_proj(x)  # [N, 1]
        iso = alpha.unsqueeze(-1) * torch.eye(3, device=x.device, dtype=x.dtype)
        # [N, 1, 1] * [3, 3] -> [N, 3, 3]

        aniso_tsr = self.tsr_proj(tsr).squeeze(-1)  # [N, 3, 3]

        # vec_proj: [N, 3, C] -> [N, 3, 2C]
        v_proj = self.vec_proj(vec)
        v1, v2 = torch.split(v_proj, v_proj.shape[-1] // 2, dim=-1)
        # v1, v2: [N, 3, C]

        #   v1.unsqueeze(2): [N, 3, 1, C]
        #   v2.unsqueeze(1): [N, 1, 3, C]
        outer_12 = v1.unsqueeze(2) * v2.unsqueeze(1)  # [N, 3, 3, C]
        outer_21 = v2.unsqueeze(2) * v1.unsqueeze(1)  # [N, 3, 3, C]
        sym_outer = outer_12 + outer_21

        aniso_vec = self.vec_out(sym_outer).squeeze(-1)  # [N, 3, 3]

        stress = iso + aniso_tsr + aniso_vec  # [N, 3, 3]
        return stress


class TensorPaiNN(nn.Module):
    """TensorPaiNN with scalar, vector, and symmetric second-order tensor channels.

    This is the full inherited architecture, with tensor message passing and
    tensor updates. It is not a head-only modification of PaiNN.
    Outputs have shapes [N,1] and [N,3,3]. The local virial has energy
    dimensions; normalization and conversion belong to the data pipeline."""

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

            self.message_layers.append(TensorPaiNNMessage(hidden_channels, num_rbf))

            self.update_layers.append(TensorPaiNNUpdate(hidden_channels))

        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_stress_feature = TensorStressOutput(hidden_channels)

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

        r_ij = positions[tgt] - positions[src] + shifts  # [E, 3]

        d_ij = torch.norm(r_ij, dim=1, keepdim=True)  # [E, 1]
        dir_ij = r_ij / (d_ij + 1e-7)  # [E, 3]

        edge_dist = d_ij.squeeze(-1)  # [E]
        edge_vector = dir_ij  # [E, 3]

        z = atomic_numbers.long()
        assert z.dim() == 1
        assert z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # [E, num_rbf]

        x = self.atom_emb(z)  # [N, C]
        vec = torch.zeros(
            x.size(0), 3, x.size(1), device=x.device, dtype=x.dtype
        )  # [N, 3, C]
        tsr = torch.zeros(
            x.size(0), 3, 3, x.size(1), device=x.device, dtype=x.dtype
        )  # [N, 3, 3, C]

        for i in range(self.num_layers):

            dx, dvec, dtsr = self.message_layers[i](
                x, vec, tsr, edge_index, edge_rbf, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            tsr = tsr + dtsr
            x = x * self.inv_sqrt_2

            dx, dvec, dtsr = self.update_layers[i](x, vec, tsr)

            x = x + dx
            vec = vec + dvec
            tsr = tsr + dtsr

        energy_per_atom = self.out_scalar_feature(x)  # [N, 1]
        stress_per_atom = self.out_stress_feature(x, vec, tsr)  # [N, 3, 3]

        return (
            energy_per_atom,
            stress_per_atom,
        )  # Both quantities have energy dimensions.

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
