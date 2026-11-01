"""Configuration and candidate records with explicit units and stable identities."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import write
import yaml

from stress_prediction.config import resolve_path as resolve_project_path

from .environment import Constraints, threshold_in_ev
from .geometry import GeometryConfig


def resolve_path(value, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty path string is required")
    return Path(resolve_project_path(value, base))


def require_keys(mapping, allowed, context):
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {context} keys: {', '.join(sorted(unknown))}")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    require_keys(config, {"schema_version", "provenance", "model", "environment", "agent", "training",
                          "evaluation", "random"}, "configuration")
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    model = config.get("model", {})
    require_keys(model, {"checkpoint", "device", "allow_legacy_pickle"}, "model")
    if "checkpoint" not in model:
        raise ValueError("model.checkpoint must identify a trained surrogate")
    model.setdefault("device", "cpu")
    model.setdefault("allow_legacy_pickle", False)
    if not isinstance(model["allow_legacy_pickle"], bool):
        raise ValueError("model.allow_legacy_pickle must be a boolean")
    env = config.get("environment", {})
    require_keys(env, {"geometry", "rotations", "vacancies", "top_k_atoms", "reward"}, "environment")
    try:
        geometry = GeometryConfig(**env.get("geometry", {}))
    except TypeError as exc:
        raise ValueError(f"Invalid environment.geometry: {exc}") from exc
    env["geometry"] = asdict(geometry)
    env.setdefault("rotations", 8)
    env.setdefault("vacancies", 9)
    env.setdefault("top_k_atoms", 100)
    Constraints(env["rotations"], env["vacancies"], geometry.n_x_grid * geometry.n_y_grid)
    positive_integer(env["top_k_atoms"], "environment.top_k_atoms")
    reward = env.get("reward", {})
    require_keys(reward, {"threshold", "scale"}, "environment.reward")
    threshold_in_ev(reward.get("threshold"))
    reward.setdefault("scale", 1.0)
    if not isinstance(reward["scale"], (int, float)) or not np.isfinite(reward["scale"]) or reward["scale"] <= 0:
        raise ValueError("environment.reward.scale must be finite and positive")
    env["reward"] = reward
    config["model"], config["environment"] = model, env
    require_keys(config.setdefault("agent", {}),
                 {"hidden_channels", "hidden_layers", "gamma", "learning_rate", "batch_size", "buffer_capacity",
                  "epsilon_start", "epsilon_end", "epsilon_decay", "target_update_frequency", "learning_starts"}, "agent")
    section_keys = {
        "training": {"episodes", "seed", "checkpoint_interval", "output_dir"},
        "evaluation": {"episodes", "seed", "temperature", "deduplicate", "max_attempts", "policy_checkpoint", "output_dir"},
        "random": {"episodes", "seed", "output_dir"},
    }
    for section, allowed in section_keys.items():
        require_keys(config.setdefault(section, {}), allowed, section)
    base = path.parent
    model["checkpoint"] = str(resolve_path(model["checkpoint"], base))
    for section in section_keys:
        for key in ("output_dir", "policy_checkpoint"):
            if key in config[section]:
                config[section][key] = str(resolve_path(config[section][key], base))
    config["config_file"] = str(path)
    return config


def write_json(path: Path, data):
    """Write portable JSON; NaN/Infinity are rejected instead of silently emitted."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_output(path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


class CandidateWriter:
    """All entries retain their ID, geometry, full tensor, and selection population."""

    def __init__(self, output: Path, method: str, config: dict, top_k_atoms: int):
        self.output = output
        (output / "structures").mkdir()
        (output / "tensors").mkdir()
        self.manifest = {"schema_version": 1, "method": method, "virial_unit": "eV",
                         "hotspot_definition": "mean_top_k_full_matrix_frobenius",
                         "top_k_atoms": top_k_atoms, "relaxation": "none",
                         "effective_config": config, "candidates": [], "status": "running"}
        self.save()

    def append(self, candidate_id, env, reward, attempt):
        if not candidate_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in candidate_id):
            raise ValueError("candidate_id must contain only letters, digits, underscores and hyphens")
        if any(row["id"] == candidate_id for row in self.manifest["candidates"]):
            raise ValueError(f"Duplicate candidate ID {candidate_id}")
        if env.last_prediction is None or not env.done:
            raise ValueError("Only a completed episode can be exported")
        atoms, energies, virial, magnitudes = env.last_prediction
        score = float(np.sort(magnitudes)[-min(env.top_k_atoms, len(atoms)):].mean())
        atoms = atoms.copy()
        atoms.info.update(candidate_id=candidate_id, virial_unit="eV", hotspot_ev=score,
                          top_k_atoms=min(env.top_k_atoms, len(atoms)), relaxation="none")
        atoms.set_array("surrogate_energy_ev", energies)
        atoms.set_array("surrogate_virial_ev", virial.reshape(-1, 9))
        atoms.set_array("hotspot_ev", magnitudes)
        structure_path = Path("structures") / f"{candidate_id}.extxyz"
        tensor_path = Path("tensors") / f"{candidate_id}.npz"
        write(self.output / structure_path, atoms, format="extxyz")
        np.savez_compressed(self.output / tensor_path, atomic_energy_ev=energies, virial_ev=virial,
                            hotspot_ev=magnitudes, atomic_numbers=atoms.numbers,
                            original_index=atoms.arrays["original_index"], positions_angstrom=atoms.positions,
                            cell_angstrom=np.asarray(atoms.cell), pbc=atoms.pbc)
        row = {"id": candidate_id, "attempt": int(attempt), "actions": list(env.actions),
               "rotations": env.rotations, "vacancies": env.vacancies, "n_atoms": len(atoms),
               "hotspot_ev": score, "reward": float(reward), "structure": structure_path.as_posix(),
               "tensors": tensor_path.as_posix()}
        self.manifest["candidates"].append(row)
        self.save()
        return row

    def save(self):
        write_json(self.output / "population.json", self.manifest)

    def finish(self, requested, attempts, status="complete", note=None):
        rows = self.manifest["candidates"]
        self.manifest.update(status=status, requested_candidates=int(requested), completed_candidates=len(rows),
                             attempts=int(attempts), unique_action_sequences=len({tuple(row["actions"]) for row in rows}),
                             note=note)
        self.save()
        return self.manifest
