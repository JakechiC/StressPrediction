"""Strict stored-force loading for the isolated E+F(+W) experiment.

No calculator is invoked. Every frame and every atom must carry finite force
labels in eV/Angstrom; the audit hashes raw float64 labels before rounding.
"""

import hashlib
from pathlib import Path
import warnings

import numpy as np
import torch
from ase.io import iread
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data

from .tensor_format import to_matrix, virial_factor_to_eV


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atoms_to_graph(atoms, cutoff, *, dtype=torch.float32, pbc=None, frame_id=0):
    """Build a directed periodic graph; respect file PBC unless explicitly overridden."""
    if cutoff <= 0 or len(atoms) == 0 or not np.isfinite(atoms.positions).all():
        raise ValueError("Graph requires a positive cutoff and finite, nonempty positions")
    work = atoms.copy()
    if pbc is not None:
        work.pbc = pbc
    source, target, shift = neighbor_list("ijS", work, cutoff)
    # Retain every atom, including atoms with no neighbors at the model cutoff.
    # The legacy dataset preserved these nodes and even entirely edgeless
    # frames. Dropping them would alter targets, frame identities and the split.
    distances = work.positions[target] - work.positions[source] + shift @ work.cell.array
    if np.any(np.linalg.norm(distances, axis=1) < 1e-8):
        raise ValueError(f"Frame {frame_id} contains coincident atoms/zero-length edges")
    return Data(
        z=torch.as_tensor(work.numbers, dtype=torch.long),
        positions=torch.as_tensor(work.positions, dtype=dtype),
        cell=torch.as_tensor(work.cell.array, dtype=dtype).unsqueeze(0),
        edge_index=torch.as_tensor(np.stack((source, target)), dtype=torch.long),
        shifts=torch.as_tensor(shift @ work.cell.array, dtype=dtype),
        frame_id=torch.full((len(work),), frame_id, dtype=torch.long),
        atom_id=torch.arange(len(work), dtype=torch.long),
        num_nodes=len(work),
    )


def stored_forces(atoms, *, key="forces", frame=0):
    """Read arrays or ASE's SinglePointCalculator results without calculation."""
    candidates = {}
    if key in atoms.arrays:
        candidates["arrays"] = np.asarray(atoms.arrays[key], dtype=np.float64)
    results = getattr(getattr(atoms, "calc", None), "results", {})
    if key in results:
        candidates["calculator_results"] = np.asarray(results[key], dtype=np.float64)
    if not candidates:
        raise ValueError(f"Frame {frame} has no stored force labels {key!r}")
    for source, values in candidates.items():
        if values.shape != (len(atoms), 3) or not np.isfinite(values).all():
            raise ValueError(f"Frame {frame} has invalid stored forces in {source}")
    if len(candidates) == 2 and not np.array_equal(*candidates.values()):
        raise ValueError(f"Frame {frame} has conflicting arrays/calculator force labels")
    return next(iter(candidates.values())), "+".join(candidates)


def read_dataset(settings, *, dtype=torch.float32, force_settings=None, return_audit=False):
    """Read raw labeled frames directly into the requested floating dtype.

    Float64 evaluation must use this entry point, not promote graphs that have
    already rounded coordinates or labels to float32. Training retains float32.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Dataset dtype must be torch.float32 or torch.float64")
    force_settings = force_settings or {"force_key": "forces", "force_unit": "eV/A"}
    if force_settings.get("force_unit") != "eV/A":
        raise ValueError("Stored force labels must have the explicitly declared unit eV/A")
    graphs = []
    force_digest = hashlib.sha256(b"stress-prediction.raw-force-labels.v1\n")
    source_counts = {}
    atom_count = 0
    factor = virial_factor_to_eV(settings.get("virial_unit", "GPa*A^3"))
    for frame, atoms in enumerate(iread(settings["path"], index=":")):
        graph = atoms_to_graph(atoms, settings["cutoff"], dtype=dtype,
                               pbc=settings.get("pbc"), frame_id=frame)
        e_key = settings.get("energy_key", "energy_per_atom")
        w_key = settings.get("virial_key", "stress_per_atom")
        if e_key not in atoms.arrays or w_key not in atoms.arrays:
            raise ValueError(f"Frame {frame} needs per-atom arrays {e_key!r} and {w_key!r}")
        energy = np.asarray(atoms.arrays[e_key]).reshape(-1)
        virial = to_matrix(atoms.arrays[w_key]) * factor
        if energy.shape != (len(atoms),) or len(virial) != len(atoms) or not np.isfinite(energy).all():
            raise ValueError(f"Invalid target shape/value in frame {frame}")
        mask = np.asarray(atoms.arrays.get(settings.get("mask_key", "representative_mask"), np.ones(len(atoms), bool)))
        if mask.shape != (len(atoms),) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid representative mask in frame {frame}")
        if not mask.any():
            raise ValueError(f"Frame {frame} has no supervised atoms")
        graph.node_energy = torch.as_tensor(energy, dtype=dtype)
        graph.node_stress = torch.as_tensor(virial, dtype=dtype)
        graph.representative_mask = torch.as_tensor(mask, dtype=torch.bool)
        forces, source = stored_forces(atoms, key=force_settings["force_key"], frame=frame)
        force_digest.update(np.asarray([frame, len(atoms)], dtype="<i8").tobytes())
        force_digest.update(np.asarray(forces, dtype="<f8").tobytes(order="C"))
        source_counts[source] = source_counts.get(source, 0) + 1
        atom_count += len(atoms)
        graph.node_force = torch.as_tensor(forces.copy(), dtype=dtype)
        graphs.append(graph)
    if len(graphs) == 0:
        raise ValueError("Dataset is empty")
    audit = {
        "schema": "raw-force-labels.v1: frame/count little-endian int64, forces float64 C-order",
        "force_labels_sha256": force_digest.hexdigest(), "force_key": force_settings["force_key"],
        "force_unit": "eV/A", "n_frames": len(graphs), "n_atoms": atom_count,
        "all_frames_have_finite_forces": True, "all_atoms_have_finite_forces": True,
        "calculator_executed": False, "source_frame_counts": source_counts,
        "coordinates_or_labels_modified": False,
    }
    return (graphs, audit) if return_audit else graphs


def validate_split(parts, count):
    keys = ("train_indices", "val_indices", "test_indices")
    if not all(key in parts for key in keys):
        raise ValueError(f"Split must contain {keys}")
    result = {}
    for key in keys:
        value = np.asarray(parts[key])
        if value.ndim != 1 or value.size == 0 or value.dtype.kind not in "iu":
            raise ValueError(f"{key} must be a nonempty integer vector")
        result[key] = value.astype(np.int64)
    all_ids = np.concatenate(list(result.values()))
    if len(all_ids) != count or not np.array_equal(np.sort(all_ids), np.arange(count)):
        raise ValueError("Split must partition every frame exactly once, without overlap or out-of-range indices")
    return result


def get_split(path, count, *, num_train=3200, num_val=400, seed=123, dataset_hash=None, create=False):
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            result = validate_split(saved, count)
            if "dataset_sha256" in saved:
                if dataset_hash is not None and str(saved["dataset_sha256"].item()) != dataset_hash:
                    raise ValueError("Dataset SHA256 differs from the dataset recorded in this split")
            else:
                warnings.warn("Legacy split has no dataset fingerprint; verify original frame order", stacklevel=2)
        return result
    if not create:
        raise FileNotFoundError(f"Split does not exist: {path}; create it during training or supply the original split")
    if not 0 < num_train < num_train + num_val < count:
        raise ValueError("Split counts must leave nonempty train, validation, and test sets")
    # RandomState preserves the legacy np.random.shuffle permutation for this seed,
    # without tying the split to model initialization or subsequent random draws.
    permutation = np.random.RandomState(seed).permutation(count)
    result = validate_split({
        "train_indices": permutation[:num_train],
        "val_indices": permutation[num_train:num_train + num_val],
        "test_indices": permutation[num_train + num_val:],
    }, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two runs silently replacing a shared split.
    with path.open("xb") as handle:
        np.savez(handle, **result, dataset_sha256=dataset_hash or "", split_seed=seed)
    return result


def energy_statistics(graphs, indices):
    """Train-only atomic energy statistics; sample SD, as in the source protocol.

    All training atoms enter normalization; representative_mask selects loss and
    validation atoms. This distinction is recorded rather than inferred.
    """
    energy = torch.cat([graphs[int(i)].node_energy.double() for i in indices])
    if energy.numel() < 2:
        raise ValueError("At least two training atom energies are needed for sample standard deviation")
    mean, std = energy.mean().item(), energy.std(correction=1).item()
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError("Training energy standard deviation is zero or too small for shared E/W normalization")
    return {"Emean": mean, "Estd": std, "normalization_atoms": "all training atoms", "std_correction": 1}
