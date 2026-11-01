"""YAML configuration with paths relative to the file that declares them."""

from copy import deepcopy
import json
import os
import math
from pathlib import Path
import re

import yaml


DEFAULTS = {
    "schema_version": 1,
    "run": {"seed": 123, "device": "auto"},
    "data": {
        "cutoff": 5.0,
        "energy_key": "energy_per_atom",
        "virial_key": "stress_per_atom",
        "virial_unit": "GPa*A^3",
        "mask_key": "representative_mask",
        "pbc": None,
    },
    "split": {"num_train": 3200, "num_val": 400, "seed": 123},
    "model": {"name": "PaiNNDirect", "parameters": {}},
    "training": {
        "epochs": 1000,
        "batch_size": 8,
        "val_batch_size": 16,
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 1e-8,
        "amsgrad": True,
        "energy_weight": 1.0,
        "virial_weight": 100.0,
        "huber_delta": 0.01,
        "gradient_clip": 10.0,
        "patience": 0,
        "min_delta": 0.0,
        "scheduler": {"period": 10, "multiplier": 1, "eta_min": 1e-6},
    },
}
PATH_FIELDS = (("run", "output_dir"), ("data", "path"), ("split", "path"))


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(value, base):
    """Expand ${VARIABLE} and ~; fail if an environment variable is missing."""
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        raise ValueError("Paths must be nonempty strings")
    def substitute(match):
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Environment variable {name!r} is not set")
        return os.environ[name]

    path = Path(re.sub(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", substitute, str(value))).expanduser()
    return str((path if path.is_absolute() else Path(base) / path).resolve())


def _read(path, stack):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    with path.open(encoding="utf-8") as handle:
        # Generated experiment configs use strict JSON. YAML 1.1 interprets
        # JSON exponent literals such as 1e-08 as strings; parse the declared
        # format so optimizer/scheduler numbers retain their intended types.
        config = (json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = config.pop("extends", None)
    base = _read(resolve_path(parent, path.parent), stack + [path]) if parent else {}
    for group, key in PATH_FIELDS:
        section = config.get(group, {})
        if not isinstance(section, dict):
            raise ValueError(f"{group} must be a mapping")
        value = section.get(key)
        if value is not None:
            config[group][key] = resolve_path(value, path.parent)
    return merge(base, config)


def load_config(path, overrides=None):
    """Read a training config. CLI path overrides are relative to current cwd."""
    config = merge(DEFAULTS, _read(path, []))
    if overrides:
        values = deepcopy(overrides)
        for group, key in PATH_FIELDS:
            section = values.get(group, {})
            if not isinstance(section, dict):
                raise ValueError(f"{group} must be a mapping")
            if section.get(key) is not None:
                values[group][key] = resolve_path(values[group][key], Path.cwd())
        config = merge(config, values)
    validate_config(config)
    return config


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Only schema_version: 1 is supported")
    for group in ("run", "data", "split", "model", "training"):
        if not isinstance(config[group], dict):
            raise ValueError(f"{group} must be a mapping")
        extra = set(config[group]) - set(DEFAULTS[group]) - {
            key for section, key in PATH_FIELDS if section == group
        }
        if extra:
            raise ValueError(f"Unknown {group} fields: {sorted(extra)}")
    for group, key in PATH_FIELDS:
        if not config[group].get(key):
            raise ValueError(f"Missing configuration field {group}.{key}")
    if not isinstance(config["run"]["device"], str) or not config["run"]["device"].strip():
        raise ValueError("run.device must be a nonempty device string")
    train = config["training"]
    def integer(value, label, minimum=0):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")

    def finite(value, label, minimum=0):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"{label} must be finite and >= {minimum}")

    for group in ("run", "split"):
        integer(config[group]["seed"], f"{group}.seed")
        if config[group]["seed"] >= 2**32:
            raise ValueError(f"{group}.seed must be < 2**32")
    for key in ("num_train", "num_val"):
        integer(config["split"][key], f"split.{key}", 1)
    for key in ("num_workers", "patience"):
        integer(train[key], f"training.{key}")
    for key in ("energy_weight", "virial_weight", "weight_decay", "min_delta"):
        finite(train[key], f"training.{key}")
    if not isinstance(train["amsgrad"], bool):
        raise ValueError("training.amsgrad must be a YAML boolean")
    for key in ("epochs", "batch_size", "val_batch_size"):
        if not isinstance(train[key], int) or isinstance(train[key], bool) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "huber_delta", "gradient_clip"):
        finite(train[key], f"training.{key}")
        if train[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if min(train["energy_weight"], train["virial_weight"]) < 0 or (
        train["energy_weight"] + train["virial_weight"] <= 0
    ):
        raise ValueError("Loss weights must be nonnegative, with at least one positive")
    if train["num_workers"] < 0 or train["patience"] < 0:
        raise ValueError("num_workers and patience must be nonnegative")
    finite(config["data"]["cutoff"], "data.cutoff")
    if config["data"]["cutoff"] <= 0:
        raise ValueError("data.cutoff must be positive")
    if config["data"]["virial_unit"] not in ("eV", "GPa*A^3", "GPa·Å³", "bar*A^3"):
        raise ValueError("virial_unit must be eV, GPa*A^3, or bar*A^3 (not pressure)")
    pbc = config["data"]["pbc"]
    if pbc is not None and (
        not isinstance(pbc, list) or len(pbc) != 3 or not all(isinstance(x, bool) for x in pbc)
    ):
        raise ValueError("data.pbc must be null or three YAML booleans")
    scheduler = train["scheduler"]
    if not isinstance(scheduler, dict):
        raise ValueError("training.scheduler must be a mapping")
    if set(scheduler) != {"period", "multiplier", "eta_min"}:
        raise ValueError("scheduler requires period, multiplier, eta_min")
    integer(scheduler["period"], "scheduler.period", 1)
    integer(scheduler["multiplier"], "scheduler.multiplier", 1)
    finite(scheduler["eta_min"], "scheduler.eta_min")
    if not isinstance(config["model"]["parameters"], dict):
        raise ValueError("model.parameters must be a mapping")
    if config["model"]["name"] not in (
        "PaiNNDirect", "PaiNNDerivative", "PaiNNEdge",
        "TensorNetDirect", "TensorNetDerivative", "TensorPaiNN",
    ):
        raise ValueError(f"Unknown model name: {config['model']['name']!r}")
    for key in ("hidden_channels", "num_layers", "num_rbf", "num_elements", "max_z"):
        if key in config["model"]["parameters"]:
            integer(config["model"]["parameters"][key], f"model.parameters.{key}", 1)
    for key in ("energy_key", "virial_key", "mask_key"):
        if not isinstance(config["data"][key], str) or not config["data"][key].strip():
            raise ValueError(f"data.{key} must be a nonempty field name")


def model_spec(config):
    """Resolve the two model families' distinct cutoff constructor names."""
    spec = deepcopy(config["model"])
    cutoff_key = "cutoff_upper" if spec["name"].startswith("TensorNet") else "cutoff"
    parameters = spec["parameters"]
    if cutoff_key in parameters and parameters[cutoff_key] != config["data"]["cutoff"]:
        raise ValueError("Model and neighbor-list cutoffs must agree")
    parameters[cutoff_key] = config["data"]["cutoff"]
    return spec
