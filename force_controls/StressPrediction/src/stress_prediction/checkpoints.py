"""Self-describing checkpoints and explicit, opt-in legacy import."""

from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
import pickle
import warnings

import numpy as np
import torch
from torch_geometric.data import Batch

from .data import atoms_to_graph
from .models import build_model

FORMAT = "stress_prediction.v1"


def select_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    return result


def primitive(value):
    if isinstance(value, Namespace):
        return {key: primitive(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    return value


def read_checkpoint(path, *, allow_legacy_pickle=False):
    """Safe tensor/primitive load by default; never silently enable unrestricted pickle."""
    try:
        result = torch.load(Path(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        if not allow_legacy_pickle:
            raise ValueError(
                "Checkpoint contains legacy pickle objects. Only for a file you trust, "
                "pass --allow-legacy-pickle (or allow_legacy_pickle=True)."
            ) from error
        warnings.warn("Loading a trusted legacy pickle checkpoint; arbitrary pickle code may execute", stacklevel=2)
        result = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(result, dict):
        raise ValueError("Expected a checkpoint dictionary, not a pickled model object")
    return result


def checkpoint_spec(checkpoint):
    """Return model, normalization, and graph conventions without guessing a model."""
    if checkpoint.get("format") == FORMAT:
        return checkpoint["model_spec"], checkpoint["stats"], checkpoint["data_spec"]
    if "format" in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint['format']!r}")
    config = primitive(checkpoint.get("config"))
    if not isinstance(config, dict) or not {"model_name", "dataset"} <= set(config):
        raise ValueError("Legacy checkpoint needs config.model_name, config.dataset, stats and model state")
    dataset = config["dataset"]
    if dataset.get("divided_by_volume", True):
        raise ValueError("Volume-normalized legacy stress checkpoints are not energy-valued virial models")
    if "cutoff" not in dataset:
        raise ValueError("Legacy checkpoint does not record a graph cutoff")
    name = config["model_name"]
    original = config.get("model", {})
    params = {key: original.get(key, default) for key, default in (
        ("hidden_channels", 128), ("num_layers", 3), ("num_rbf", 64)
    )}
    if name.startswith("TensorNet"):
        params.update(cutoff_upper=dataset["cutoff"], max_z=config.get("num_elements", 86))
        for key, default in (("rbf_type", "expnorm"), ("trainable_rbf", False), ("activation", "silu")):
            params[key] = original.get(key, default)
    else:
        params.update(cutoff=dataset["cutoff"], num_elements=config.get("num_elements", 86))
    data_spec = {
        "cutoff": dataset["cutoff"], "pbc": [True, True, True],
        "energy_key": "energy_per_atom", "virial_key": "stress_per_atom",
        "virial_unit": "GPa*A^3", "mask_key": "representative_mask",
    }
    warnings.warn(
        "Legacy model imported using the original factory defaults and all-axis PBC. "
        "Verify this checkpoint's source version; import is not a claim of manuscript reproduction.",
        stacklevel=2,
    )
    return {"name": name, "parameters": params}, checkpoint["stats"], data_spec


def validate_stats(stats):
    mean, std = float(stats["Emean"]), float(stats["Estd"])
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Checkpoint energy normalization must be finite with positive Estd")
    return mean, std


def load_model_weights(model, weights):
    """Require the recorded architecture to match every checkpoint parameter."""
    if not isinstance(weights, Mapping):
        raise ValueError("Checkpoint has no valid model state dictionary")
    try:
        model.load_state_dict(weights, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "Checkpoint weights do not match the recorded model architecture. "
            "Recover the matching source configuration; layer counts and heads "
            "are not guessed or silently changed. " + str(error)
        ) from error


class Predictor:
    """Callable ASE Atoms -> (atomic energy eV[N], virial eV[N,3,3])."""

    def __init__(self, model, stats, data_spec, device="cpu"):
        self.device = select_device(device)
        self.model = model.to(self.device).eval()
        self.mean, self.std = validate_stats(stats)
        self.data_spec = dict(data_spec)
        self.dtype = next(model.parameters()).dtype

    def __call__(self, atoms):
        graph = atoms_to_graph(
            atoms, self.data_spec["cutoff"], dtype=self.dtype, pbc=self.data_spec.get("pbc")
        )
        graph = Batch.from_data_list([graph]).to(self.device)
        # Derivative forwards enable their own required autograd graph. Inference
        # mode would disable this; no_grad is deliberately used instead.
        with torch.no_grad():
            energy, virial = self.model(graph)
        return (
            energy.detach().cpu().numpy().reshape(-1) * self.std + self.mean,
            virial.detach().cpu().numpy() * self.std,
        )


def load_predictor(checkpoint, device="cpu", allow_legacy_pickle=False):
    state = read_checkpoint(checkpoint, allow_legacy_pickle=allow_legacy_pickle)
    spec, stats, data_spec = checkpoint_spec(state)
    model = build_model(spec["name"], spec["parameters"])
    weights = state.get("model_state", state.get("model"))
    if weights is None:
        raise ValueError("Checkpoint has no model state dictionary")
    dtype = next(value.dtype for value in weights.values() if torch.is_floating_point(value))
    model.to(dtype=dtype)
    load_model_weights(model, weights)
    return Predictor(model, stats, data_spec, device)


def atomic_save(state, path):
    """Replace a checkpoint only once serialization has completed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
