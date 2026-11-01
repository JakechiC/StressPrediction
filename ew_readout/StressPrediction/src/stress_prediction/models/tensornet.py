# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from typing import Tuple
from torch import Tensor, nn
from .tensornet_utils import (
    CosineCutoff,
    rbf_class_mapping,
    act_class_mapping,
)
from torch_scatter import scatter
from torch.autograd import grad


__all__ = ["TensorNetDirect", "TensorNetDerivative"]


def vector_to_skewtensor(vector):
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector):
    """Creates a symmetric traceless tensor from the outer product of a vector with itself."""
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


def decompose_tensor(tensor):
    """Full tensor decomposition into irreducible components."""
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return I, A, S


def tensor_norm(tensor):
    """Compute the squared Frobenius norm used by the inherited architecture."""
    return (tensor**2).sum((-2, -1))


class TensorNetDirect(nn.Module):
    """TensorNet with invariant-gated isotropic and symmetric tensor readouts."""

    def __init__(
        self,
        hidden_channels=128,
        num_layers=3,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_upper=5.0,
        max_z=86,
        dtype=torch.float32,
    ):
        super(TensorNetDirect, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_upper = cutoff_upper
        act_class = act_class_mapping[activation]
        self.distance_expansion = rbf_class_mapping[rbf_type](
            0, cutoff_upper, num_rbf, trainable_rbf
        )
        self.tensor_embedding = TensorEmbedding(
            hidden_channels,
            num_rbf,
            act_class,
            0,
            cutoff_upper,
            max_z,
            dtype,
        )

        self.layers = nn.ModuleList()
        if num_layers != 0:
            for _ in range(num_layers):
                self.layers.append(
                    Interaction(
                        num_rbf,
                        hidden_channels,
                        act_class,
                        0,
                        cutoff_upper,
                        dtype,
                    )
                )

        # Output projection for scalar representation
        self.scalar_proj = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)

        # I_hat = W_I * I, S_hat = W_S * S
        self.I_out_lin = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.S_out_lin = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype),
            act_class(),
            nn.Linear(hidden_channels, 2 * hidden_channels, dtype=dtype),
        )

        self.cutoff = cutoff_upper

        # --- OUTPUT HEAD ---
        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.out_stress_feature = nn.Linear(hidden_channels, 1, bias=False)
        # Keep all inherited modules and buffers consistent with constructor dtype.
        self.to(dtype=dtype)

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.scalar_proj.reset_parameters()
        self.I_out_lin.reset_parameters()
        self.S_out_lin.reset_parameters()
        for module in self.weight_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, data) -> tuple[Tensor, Tensor]:

        positions = data.positions
        edge_index = data.edge_index
        shifts = data.shifts
        num_atoms = data.num_nodes

        src, tgt = edge_index

        z = data.z

        # compute atom and pair features
        edge_vec = positions[tgt] - positions[src] + shifts

        edge_weight = edge_vec.norm(dim=-1)

        # Radial basis functions
        edge_attr = self.distance_expansion(edge_weight)

        # Normalize edge vectors (handle division by zero)
        mask = edge_index[0] == edge_index[1]
        edge_vec_norm = edge_vec / edge_weight.masked_fill(mask, 1).unsqueeze(1)

        # Tensor embedding
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec_norm, edge_attr)

        # Interaction layers (SO(3) equivariant, no charge dependence)
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr)

        # I: (N, C, 3, 3), A: (N, C, 3, 3), S: (N, C, 3, 3)
        I, A, S = decompose_tensor(X)

        # shape: (N, C)
        norm_I = tensor_norm(I)
        norm_A = tensor_norm(A)
        norm_S = tensor_norm(S)

        # shape: (N, 3*C) -> (N, 2*C)
        invariants = torch.cat([norm_I, norm_A, norm_S], dim=-1)
        weights = self.weight_mlp(invariants)
        w_I, w_S = weights.chunk(2, dim=-1)  # (N, C)

        # (N, C, 3, 3) -> (N, 3, 3, C)
        I_proj = self.I_out_lin(I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S_proj = self.S_out_lin(S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        I_final = w_I.unsqueeze(-1).unsqueeze(-1) * I_proj  # (N, C, 3, 3)
        S_final = w_S.unsqueeze(-1).unsqueeze(-1) * S_proj

        atomic_tensor_output = I_final + S_final

        scalar_rep = self.scalar_proj(invariants)

        #### Output block #####################################################

        energy_per_atom = self.out_scalar_feature(scalar_rep)  # [N, 1]
        stress_per_atom = (
            atomic_tensor_output.permute(0, 2, 3, 1)
            @ self.out_stress_feature.weight.t()
        ).squeeze(
            -1
        )  # [N, 3, 3]

        return (
            energy_per_atom,
            stress_per_atom,
        )  # Model-space energy and virial; both have energy dimensions.

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TensorNetDerivative(nn.Module):
    """TensorNet virials from energy derivatives with inherited local allocation.

    Evaluation requires autograd; use no_grad(), not inference_mode().
    """

    def __init__(
        self,
        hidden_channels=128,
        num_layers=3,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        cutoff_upper=5.0,
        max_z=86,
        dtype=torch.float32,
    ):
        super(TensorNetDerivative, self).__init__()

        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff_upper = cutoff_upper
        act_class = act_class_mapping[activation]
        self.distance_expansion = rbf_class_mapping[rbf_type](
            0, cutoff_upper, num_rbf, trainable_rbf
        )
        self.tensor_embedding = TensorEmbedding(
            hidden_channels,
            num_rbf,
            act_class,
            0,
            cutoff_upper,
            max_z,
            dtype,
        )

        self.layers = nn.ModuleList()
        if num_layers != 0:
            for _ in range(num_layers):
                self.layers.append(
                    Interaction(
                        num_rbf,
                        hidden_channels,
                        act_class,
                        0,
                        cutoff_upper,
                        dtype,
                    )
                )

        # Output projection for scalar representation
        self.scalar_proj = nn.Linear(3 * hidden_channels, hidden_channels, dtype=dtype)

        self.cutoff = cutoff_upper

        # --- OUTPUT HEAD ---
        self.out_scalar_feature = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.to(dtype=dtype)

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.scalar_proj.reset_parameters()

    @torch.enable_grad()
    def forward(self, data) -> tuple[Tensor, Tensor]:

        positions = data.positions
        edge_index = data.edge_index
        shifts = data.shifts
        num_atoms = data.num_nodes

        src, tgt = edge_index

        z = data.z

        # compute atom and pair features
        edge_vec = positions[tgt] - positions[src] + shifts
        edge_vec = edge_vec.requires_grad_(True)  # for edge forces

        edge_weight = edge_vec.norm(dim=-1)

        # Radial basis functions
        edge_attr = self.distance_expansion(edge_weight)

        # Normalize edge vectors (handle division by zero)
        mask = edge_index[0] == edge_index[1]
        edge_vec_norm = edge_vec / edge_weight.masked_fill(mask, 1).unsqueeze(1)

        # Tensor embedding
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec_norm, edge_attr)

        # Interaction layers (SO(3) equivariant, no charge dependence)
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr)

        # I: (N, C, 3, 3), A: (N, C, 3, 3), S: (N, C, 3, 3)
        I, A, S = decompose_tensor(X)

        # shape: (N, C)
        norm_I = tensor_norm(I)
        norm_A = tensor_norm(A)
        norm_S = tensor_norm(S)

        # shape: (N, 3*C) -> (N, 2*C)
        invariants = torch.cat([norm_I, norm_A, norm_S], dim=-1)

        scalar_rep = self.scalar_proj(invariants)

        #### Output block #####################################################

        energy_per_atom = self.out_scalar_feature(scalar_rep)  # [N, 1]
        energy = scatter(energy_per_atom, data.batch, 0, dim_size=data.num_graphs)

        # edge forces
        dEdrij = grad(
            energy,
            edge_vec,
            grad_outputs=torch.ones_like(energy),
            create_graph=self.training,
        )[0]

        if dEdrij is None:
            edge_forces = torch.zeros_like(edge_vec)
        else:
            edge_forces = -1 * dEdrij

        edge_virial = torch.einsum("zi,zj->zij", edge_forces, edge_vec)
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


class TensorEmbedding(nn.Module):
    """Tensor embedding layer.

    :meta private:
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        activation,
        cutoff_lower,
        cutoff_upper,
        max_z=128,
        dtype=torch.float32,
    ):
        super(TensorEmbedding, self).__init__()

        self.hidden_channels = hidden_channels
        self.distance_proj1 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj2 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.distance_proj3 = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.max_z = max_z
        self.emb = nn.Embedding(max_z, hidden_channels, dtype=dtype)
        self.emb2 = nn.Linear(2 * hidden_channels, hidden_channels, dtype=dtype)
        self.act = activation()
        self.linears_tensor = nn.ModuleList()
        for _ in range(3):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _get_atomic_number_message(self, z: Tensor, edge_index: Tensor) -> Tensor:
        Z = self.emb(z)
        Zij = self.emb2(
            Z.index_select(0, edge_index.t().reshape(-1)).view(
                -1, self.hidden_channels * 2
            )
        )[..., None, None]
        return Zij

    def _get_tensor_messages(
        self, Zij: Tensor, edge_weight: Tensor, edge_vec_norm: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        C = self.cutoff(edge_weight).reshape(-1, 1, 1, 1) * Zij
        eye = torch.eye(3, 3, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)[
            None, None, ...
        ]
        Iij = self.distance_proj1(edge_attr)[..., None, None] * C * eye
        Aij = (
            self.distance_proj2(edge_attr)[..., None, None]
            * C
            * vector_to_skewtensor(edge_vec_norm)[..., None, :, :]
        )
        Sij = (
            self.distance_proj3(edge_attr)[..., None, None]
            * C
            * vector_to_symtensor(edge_vec_norm)[..., None, :, :]
        )
        return Iij, Aij, Sij

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec_norm: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        Zij = self._get_atomic_number_message(z, edge_index)
        Iij, Aij, Sij = self._get_tensor_messages(
            Zij, edge_weight, edge_vec_norm, edge_attr
        )
        source = torch.zeros(
            z.shape[0], self.hidden_channels, 3, 3, device=z.device, dtype=Iij.dtype
        )
        I = source.index_add(dim=0, index=edge_index[0], source=Iij)
        A = source.index_add(dim=0, index=edge_index[0], source=Aij)
        S = source.index_add(dim=0, index=edge_index[0], source=Sij)
        norm = self.init_norm(tensor_norm(I + A + S))
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(-1, self.hidden_channels, 3)
        I = (
            self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 0, None, None]
        )
        A = (
            self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 1, None, None]
        )
        S = (
            self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            * norm[..., 2, None, None]
        )
        X = I + A + S
        return X


def tensor_message_passing(
    edge_index: Tensor, factor: Tensor, tensor: Tensor, natoms: int
) -> Tensor:
    """Message passing for tensors."""
    msg = factor * tensor.index_select(0, edge_index[1])
    shape = (natoms, tensor.shape[1], tensor.shape[2], tensor.shape[3])
    tensor_m = torch.zeros(*shape, device=tensor.device, dtype=tensor.dtype)
    tensor_m = tensor_m.index_add(0, edge_index[0], msg)
    return tensor_m


class Interaction(nn.Module):
    """Interaction layer.

    :meta private:
    """

    def __init__(
        self,
        num_rbf,
        hidden_channels,
        activation,
        cutoff_lower,
        cutoff_upper,
        dtype=torch.float32,
    ):
        super(Interaction, self).__init__()

        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(
            nn.Linear(num_rbf, hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_scalar.append(
            nn.Linear(2 * hidden_channels, 3 * hidden_channels, bias=True, dtype=dtype)
        )
        self.linears_tensor = nn.ModuleList()
        for _ in range(6):
            self.linears_tensor.append(
                nn.Linear(hidden_channels, hidden_channels, bias=False)
            )
        self.act = activation()
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            linear.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()

    def forward(
        self,
        X: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        C = self.cutoff(edge_weight)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(
            edge_attr.shape[0], self.hidden_channels, 3
        )
        X = X / (tensor_norm(X) + 1)[..., None, None]
        I, A, S = decompose_tensor(X)
        I = self.linears_tensor[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = I + A + S
        Im = tensor_message_passing(
            edge_index, edge_attr[..., 0, None, None], I, X.shape[0]
        )
        Am = tensor_message_passing(
            edge_index, edge_attr[..., 1, None, None], A, X.shape[0]
        )
        Sm = tensor_message_passing(
            edge_index, edge_attr[..., 2, None, None], S, X.shape[0]
        )
        msg = Im + Am + Sm

        B = torch.matmul(Y, msg)
        I, A, S = decompose_tensor(2 * B)

        normp1 = (tensor_norm(I + A + S) + 1)[..., None, None]
        I, A, S = I / normp1, A / normp1, S / normp1
        I = self.linears_tensor[3](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.linears_tensor[4](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.linears_tensor[5](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        dX = I + A + S
        X = X + dX + torch.matrix_power(dX, 2)
        return X
