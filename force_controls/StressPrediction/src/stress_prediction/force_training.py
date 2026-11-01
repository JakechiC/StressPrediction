"""Isolated E+F(+W) training; execute only on an allocated compute node.

Original model architectures and E/W training are unchanged. Forces are the
negative coordinate gradient of the sum of normalized atomic energies, with
the historical train-only energy standard deviation and explicit length scale.
"""

from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from . import __version__
from .checkpoints import FORMAT, atomic_save, read_checkpoint, select_device, validate_stats
from .config import DEFAULTS, PATH_FIELDS, _read, merge, model_spec, resolve_path, validate_config
from .data import energy_statistics, file_digest, get_split, validate_split
from .force_data import read_dataset
from .models import build_model


FORCE_DEFAULTS = {"force_weight": 1.0, "force_length_A": 1.0,
                  "force_key": "forces", "force_unit": "eV/A"}


def validate_force_config(config):
    """Validate the original schema separately from explicit force settings."""
    base = deepcopy(config)
    supplied = base.pop("force", None)
    if not isinstance(supplied, dict) or set(supplied) != set(FORCE_DEFAULTS):
        raise ValueError("force must explicitly specify force_weight, force_length_A, force_key, force_unit")
    ordinary = deepcopy(base)
    ordinary["training"].pop("microbatch_size", None)
    ordinary["training"].pop("evaluation_batch_size", None)
    validate_config(ordinary)
    for name, default in (("microbatch_size", base["training"]["batch_size"]),
                          ("evaluation_batch_size", 1)):
        value = base["training"].get(name, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if base["training"].get("microbatch_size", base["training"]["batch_size"]) > base["training"]["batch_size"]:
        raise ValueError("microbatch_size cannot exceed logical batch_size")
    if base["model"]["name"] not in ("PaiNNDerivative", "TensorNetDerivative"):
        raise ValueError("This force experiment supports the two frozen derivative models only")
    for key in ("force_weight", "force_length_A"):
        value = supplied[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"force.{key} must be finite and positive")
    if not isinstance(supplied["force_key"], str) or not supplied["force_key"].strip():
        raise ValueError("force.force_key must name a stored label")
    if supplied["force_unit"] != "eV/A":
        raise ValueError("force.force_unit must be eV/A")
    return dict(supplied)


def load_config(path, overrides=None):
    """Load inherited force YAML without extending the original E/W schema."""
    config = merge(DEFAULTS, _read(path, []))
    if overrides:
        values = deepcopy(overrides)
        for group, key in PATH_FIELDS:
            section = values.get(group, {})
            if not isinstance(section, dict):
                raise ValueError(f"{group} must be a mapping")
            if section.get(key) is not None:
                values[group][key] = resolve_path(section[key], Path.cwd())
        config = merge(config, values)
    validate_force_config(config)
    return config


def forward_forces(model, batch, *, create_graph):
    """Return (E_norm, W_norm, -d(sum E_norm)/dR_A).

The existing derivative forwards retain their energy graph only in training
mode. They contain no dropout/BN. We temporarily enable that mode, explicitly
reject stateful training-only modules, and restore every module's mode. This
does not update model parameters/buffers or enable validation backpropagation.
The final coordinate derivative uses create_graph=False for evaluation, and
periodic shifts stay constant (fixed cell and fixed neighbor topology).
"""
    if torch.is_inference_mode_enabled():
        raise ValueError("Force evaluation cannot run inside torch.inference_mode")
    forbidden = (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)
    if any(isinstance(module, forbidden) for module in model.modules()):
        raise ValueError("Force mode bridge requires models without dropout or batch normalization")
    modes = [(module, module.training) for module in model.modules()]
    batch.positions = batch.positions.detach().requires_grad_(True)
    batch.shifts = batch.shifts.detach()
    try:
        model.train(True)
        with torch.enable_grad():
            energy, virial = model(batch)
            gradient = torch.autograd.grad(energy.sum(), batch.positions,
                                           create_graph=create_graph, retain_graph=create_graph,
                                           allow_unused=True)[0]
            # An edgeless graph can have a position-independent atomic energy.
            force = torch.zeros_like(batch.positions) if gradient is None else -gradient
    finally:
        for module, mode in modes:
            module.training = mode
    if not create_graph:
        return energy.detach(), virial.detach(), force.detach()
    return energy, virial, force


class BatchOrderFingerprint:
    """Hash the batches actually yielded, without sampling or consuming RNG.

    v1 schema: explicit little-endian int64 counts/IDs, original per-atom
    frame/atom identities, and uint8 supervision masks; batch boundaries are
    included. The frame hash omits boundaries and masks to isolate permutation.
    Update on CPU before moving the already yielded batch to the accelerator.
    """

    def __init__(self):
        self.batch_digest = hashlib.sha256(b"stress-prediction.batch-order.v1\n")
        self.frame_digest = hashlib.sha256(b"stress-prediction.frame-order.v1\n")
        self.batches = self.frames = self.atoms = 0

    def update(self, batch):
        frame_ids = batch.frame_id.detach().cpu().numpy().astype("<i8", copy=False)
        atom_ids = batch.atom_id.detach().cpu().numpy().astype("<i8", copy=False)
        mask = batch.representative_mask.detach().cpu().numpy().astype("u1", copy=False)
        starts = batch.ptr[:-1].detach().cpu().numpy()
        ordered_frames = frame_ids[starts]
        counts = np.asarray([len(ordered_frames), len(frame_ids)], dtype="<i8")
        for value in (counts, ordered_frames, frame_ids, atom_ids, mask):
            self.batch_digest.update(value.tobytes(order="C"))
        self.frame_digest.update(ordered_frames.tobytes(order="C"))
        self.batches += 1
        self.frames += len(ordered_frames)
        self.atoms += len(frame_ids)

    def record(self):
        return {
            "train_batch_order_sha256": self.batch_digest.hexdigest(),
            "train_frame_order_sha256": self.frame_digest.hexdigest(),
            "train_batches": self.batches, "train_frames": self.frames,
            "train_atoms": self.atoms,
        }


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _numpy_rng_state():
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {"name": name, "keys": keys.tolist(), "position": position,
            "has_gauss": has_gauss, "cached_gaussian": cached_gaussian}


def _restore_numpy_rng(state):
    np.random.set_state((state["name"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))


def _initial_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _prepare_batch(batch, device, dtype, *, non_blocking=False):
    # The loader supplies CPU tensors. Compute the mask count once before the
    # transfer, instead of synchronizing CUDA for repeated scalar conversions.
    count = int(batch.representative_mask.sum())
    if count <= 0:
        raise ValueError("Batch contains no supervised atoms")
    batch = batch.to(device, non_blocking=non_blocking)
    batch = batch.apply(lambda value: value.to(dtype=dtype) if torch.is_floating_point(value) else value)
    object.__setattr__(batch, "_supervised_count", count)
    return batch, count


def _loss_averages(observations, counts):
    # One transfer per epoch for the scalar diagnostics. Python's sequential
    # double-precision additions preserve the earlier weighted-loss reduction.
    rows = torch.stack(observations).detach().cpu().numpy()
    totals = [0.0, 0.0, 0.0, 0.0]
    for row, count in zip(rows, counts):
        for column in range(4):
            totals[column] += float(row[column]) * count
    return [value / sum(counts) for value in totals], rows


def backward_logical_batch(model, cpu_batch, stats, settings, force_settings, device, *, non_blocking=False):
    """Accumulate one logical batch's atom-weighted loss before one optimizer step.

    Frame order and logical boundaries stay unchanged. Each microbatch contains
    complete graphs; no gradients couple different graphs. The normalization is
    selected atom count, including the final partial logical batch.
    """
    total_count = int(cpu_batch.representative_mask.sum())
    if total_count <= 0:
        raise ValueError("Logical batch contains no supervised atoms")
    size = settings.get("microbatch_size", settings["batch_size"])
    graphs = cpu_batch.to_data_list()
    dtype = next(model.parameters()).dtype
    totals = torch.zeros(4, device=device, dtype=dtype)
    for start in range(0, len(graphs), size):
        micro = Batch.from_data_list(graphs[start:start + size])
        micro, count = _prepare_batch(micro, device, dtype, non_blocking=non_blocking)
        energy, virial, force = forward_forces(model, micro, create_graph=True)
        loss, e_loss, w_loss, f_loss = task_loss(energy, virial, force, micro, stats, settings, force_settings)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite microbatch loss")
        fraction = count / total_count
        (loss * fraction).backward()
        totals += torch.stack((loss.detach(), e_loss.detach(), w_loss.detach(), f_loss.detach())) * fraction
        del energy, virial, force, loss, e_loss, w_loss, f_loss, micro
    return totals, total_count


def _append_csv(path, row):
    new_file = not path.exists()
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle), None) != list(row):
                raise ValueError(f"CSV schema differs from this runner: {path}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _save_immutable_checkpoint(state, path):
    """Publish a completed milestone with no overwrite, even on a racing call."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
        os.link(temporary, path)  # Atomic exclusive creation on the same filesystem.
    finally:
        temporary.unlink(missing_ok=True)


def task_loss(energy, virial, force, batch, stats, settings, force_settings):
    mean, std = validate_stats(stats)
    mask = batch.representative_mask
    selected_count = getattr(batch, "_supervised_count", None)
    if (selected_count is not None and selected_count <= 0) or (selected_count is None and not mask.any()):
        raise ValueError("Batch contains no supervised atoms")
    criterion = torch.nn.HuberLoss(delta=settings["huber_delta"], reduction="mean")
    e_loss = criterion(energy.reshape(-1)[mask], ((batch.node_energy - mean) / std)[mask])
    # Keep the original full matrix loss. Symmetric off-diagonal entries occur twice.
    # W=0 must not evaluate a virial loss or attach its graph to the objective.
    # The zero reported in diagnostics means inactive, not a measured W error.
    if settings["virial_weight"] > 0:
        w_loss = criterion(virial[mask], (batch.node_stress / std)[mask])
    else:
        w_loss = e_loss.new_zeros(())
    length = force_settings["force_length_A"]
    f_loss = criterion(force[mask] * length, batch.node_force[mask] * (length / std))
    total = settings["energy_weight"] * e_loss + force_settings["force_weight"] * f_loss
    if settings["virial_weight"] > 0:
        total = total + settings["virial_weight"] * w_loss
    return total, e_loss, w_loss, f_loss


def validation_loss(model, loader, stats, settings, device, *, force_settings, non_blocking=False):
    """Compute only the atom-weighted losses needed for checkpoint selection."""
    model.eval()
    observations, counts = [], []
    dtype = next(model.parameters()).dtype
    for batch in loader:
        batch, count = _prepare_batch(batch, device, dtype, non_blocking=non_blocking)
        energy, virial, force = forward_forces(model, batch, create_graph=False)
        loss, e_loss, w_loss, f_loss = task_loss(energy, virial, force, batch, stats, settings, force_settings)
        values = (loss, e_loss, w_loss, f_loss)
        observations.append(torch.stack(tuple(value.detach() for value in values)))
        counts.append(count)
    if not counts or sum(counts) == 0:
        raise ValueError("Validation split has no selected atoms")
    averaged_losses, _ = _loss_averages(observations, counts)
    if not np.isfinite(averaged_losses).all():
        raise ValueError("Validation produced a non-finite normalized loss")
    result = {"normalized_weighted_loss": averaged_losses[0],
              "normalized_energy_loss": averaged_losses[1],
              "normalized_virial_loss": averaged_losses[2]}
    result["normalized_force_loss"] = averaged_losses[3]
    return result


def _loader(graphs, indices, settings, *, training=False, generator=None, pin_memory=False):
    return DataLoader(
        [graphs[int(i)] for i in indices],
        batch_size=settings["batch_size" if training else "val_batch_size"],
        shuffle=training,
        num_workers=settings["num_workers"],
        generator=generator,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
    )


def train(config, *, resume=None, milestone_epochs=(), checkpoint_interval=1, pin_memory=False):
    """Start a training run only when called explicitly by the user.

    Resume is supported for native epoch-boundary checkpoints. Extending the
    epoch count and changing device/output directory is allowed; changing the
    model, data, loss, optimizer or batch protocol requires a new run.
    """
    if not isinstance(pin_memory, bool):
        raise ValueError("pin_memory must be a boolean")
    if type(checkpoint_interval) is not int or checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be a positive integer")
    milestones = tuple(sorted(set(milestone_epochs)))
    if any(type(value) is not int or value < 1 for value in milestones):
        raise ValueError("milestone_epochs must contain positive one-based epoch counts")
    execution_options = {"milestone_epochs": list(milestones),
                         "checkpoint_interval": checkpoint_interval, "pin_memory": pin_memory}
    config = deepcopy(config)
    force_settings = validate_force_config(config)
    settings = config["training"]
    output = Path(config["run"]["output_dir"])
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and (resume is None or Path(resume).resolve().parent != output.resolve()):
        raise FileExistsError(f"Run directory is not empty: {output}; resume its checkpoint or use a new directory")
    device = select_device(config["run"]["device"])
    seed_everything(config["run"]["seed"])
    dataset_hash = file_digest(config["data"]["path"])
    graphs, force_audit = read_dataset(config["data"], force_settings=force_settings, return_audit=True)
    split_settings = config["split"]
    parts = get_split(
        split_settings["path"], len(graphs), num_train=split_settings["num_train"],
        num_val=split_settings["num_val"], seed=split_settings["seed"], dataset_hash=dataset_hash, create=True,
    )
    stats = energy_statistics(graphs, parts["train_indices"])
    spec = model_spec(config)
    model = build_model(spec["name"], spec["parameters"]).to(device=device, dtype=torch.float32)
    initial_hash = _initial_state_hash(model) if resume is None else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        amsgrad=settings["amsgrad"], betas=(0.9, 0.999), eps=1e-8,
    )
    schedule = settings["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=schedule["period"], T_mult=schedule["multiplier"], eta_min=schedule["eta_min"],
    )
    generator = torch.Generator().manual_seed(config["run"]["seed"])
    first_epoch, best_loss, bad_epochs, best_epoch = 0, float("inf"), 0, -1
    best_weights = None
    if resume is not None:
        saved = read_checkpoint(resume)
        if saved.get("format") != FORMAT or "optimizer_state" not in saved:
            raise ValueError("Resume requires a native training checkpoint; legacy models support prediction only")
        if saved.get("force_settings") != force_settings or saved.get("force_label_audit") != force_audit:
            raise ValueError("Resume force protocol or force labels differ from saved run")
        old_settings = dict(saved["config"]["training"])
        new_settings = dict(settings)
        old_settings.pop("epochs"); new_settings.pop("epochs")
        old_data = {k: v for k, v in saved["data_spec"].items() if k != "path"}
        new_data = {k: v for k, v in config["data"].items() if k != "path"}
        if (saved["dataset_sha256"] != dataset_hash or saved["model_spec"] != spec
                or old_settings != new_settings or old_data != new_data
                or saved["config"]["run"]["seed"] != config["run"]["seed"]):
            raise ValueError("Resume model/data/seed/training protocol differs from saved run")
        old_parts = validate_split({key: value.numpy() for key, value in saved["split"].items()}, len(graphs))
        if not all(np.array_equal(parts[key], old_parts[key]) for key in parts):
            raise ValueError("Resume split differs from saved run")
        stats = saved["stats"]
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        first_epoch = saved["epoch"] + 1
        best_loss, bad_epochs = saved["best_val_loss"], saved["bad_epochs"]
        best_epoch = saved["best_epoch"]
        best_weights = saved["best_model_state"]
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
        if "numpy_rng" in saved:
            _restore_numpy_rng(saved["numpy_rng"])
        if first_epoch >= settings["epochs"]:
            raise ValueError("Checkpoint already reached requested epochs; increase training.epochs")
    history = output / "history.csv"
    order_history = output / "train_order.csv"
    diagnostics = output / "diagnostics.csv"
    if resume is not None and history.exists():
        with history.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        if existing and int(existing[-1]["epoch"]) != first_epoch - 1:
            raise ValueError("History does not end at resume epoch; resume into a new empty output directory")
    if resume is not None and order_history.exists():
        with order_history.open(newline="", encoding="utf-8") as handle:
            existing_order = list(csv.DictReader(handle))
        if existing_order and int(existing_order[-1]["epoch"]) != first_epoch - 1:
            raise ValueError("Batch-order history does not end at resume epoch")
    if resume is not None and diagnostics.exists():
        with diagnostics.open(newline="", encoding="utf-8") as handle:
            existing_diagnostics = list(csv.DictReader(handle))
        if existing_diagnostics and int(existing_diagnostics[-1]["epoch_zero_based"]) != first_epoch - 1:
            raise ValueError("Diagnostics history does not end at resume epoch")
    if resume is not None and nonempty:
        provenance_path = output / "provenance.json"
        if not provenance_path.exists():
            raise ValueError("Nonempty resume directory has no native provenance record")
        with provenance_path.open(encoding="utf-8") as handle:
            provenance = json.load(handle)
        if provenance["dataset_sha256"] != dataset_hash:
            raise ValueError("Resume output provenance belongs to a different dataset")
    initialization = {"run_seed": config["run"]["seed"], "model_spec": spec,
                      "normalization": stats, "dataset_sha256": dataset_hash,
                      "force_settings": force_settings, "force_label_audit": force_audit,
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "initial_state_sha256": initial_hash, "hash_does_not_sample_rng": True}
    if resume is not None:
        initial_path = Path(resume).parent / "initialization.json"
        if initial_path.exists():
            with initial_path.open(encoding="utf-8") as handle:
                previous_initial = json.load(handle)
            for key in ("run_seed", "model_spec", "normalization", "dataset_sha256", "parameter_count",
                        "force_settings", "force_label_audit"):
                if previous_initial.get(key) != initialization[key]:
                    raise ValueError(f"Resume initialization differs: {key}")
            initial_hash = previous_initial.get("initial_state_sha256")
            if saved.get("initial_state_sha256", initial_hash) != initial_hash:
                raise ValueError("Checkpoint and initialization hashes differ")
            initialization = previous_initial
        else:
            initialization["initial_state_sha256"] = saved.get("initial_state_sha256")
            initialization["initialization_status"] = "unavailable for legacy checkpoint" if not initialization["initial_state_sha256"] else "recovered from checkpoint"
            initial_hash = initialization["initial_state_sha256"]
    ledger_path = output / "milestones.json"
    ledger = {"schema_version": 1, "dataset_sha256": dataset_hash, "model_spec": spec,
              "run_seed": config["run"]["seed"], "checkpoints": {}}
    if ledger_path.exists():
        with ledger_path.open(encoding="utf-8") as handle:
            prior_ledger = json.load(handle)
        if any(prior_ledger.get(key) != ledger[key] for key in ("schema_version", "dataset_sha256", "model_spec", "run_seed")):
            raise ValueError("Milestone ledger belongs to a different run")
        ledger = prior_ledger
        for completed, entry in ledger["checkpoints"].items():
            expected_name = f"model_epoch_{int(completed):06d}.ckpt"
            if entry["path"] != expected_name or file_digest(output / expected_name) != entry["sha256"]:
                raise ValueError("Immutable milestone checkpoint fingerprint mismatch")
    for completed in milestones:
        if completed <= first_epoch and str(completed) not in ledger["checkpoints"]:
            raise ValueError(f"Cannot retroactively create missing milestone {completed}")
        if completed > first_epoch and (output / f"model_epoch_{completed:06d}.ckpt").exists():
            raise FileExistsError(f"Milestone {completed} already exists; it will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "initialization.json", initialization)
    write_json(output / "config.resolved.json", config)
    write_json(output / "provenance.json", {
        "package_version": __version__, "dataset_sha256": dataset_hash,
        "torch_version": str(torch.__version__), "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "stats": stats, "split_sizes": {key: len(value) for key, value in parts.items()},
        "training_dtype": "float32", "validation_dtype": "float32",
        "logical_batch_size": settings["batch_size"],
        "microbatch_size": settings.get("microbatch_size", settings["batch_size"]),
        "microbatch_reduction": "eligible-atom weighted; one clipping and optimizer step per logical batch",
        "optimizer_fixed_defaults": {"betas": [0.9, 0.999], "eps": 1e-8},
        "training_order": {"path": "train_order.csv", "schema": "v1 little-endian IDs, masks and batch boundaries",
                           "rng_consumed_by_audit": False},
        "execution_options": execution_options,
        "force_settings": force_settings, "force_label_audit": force_audit,
        "force_definition": "F_eV_A = -Estd * d(sum atomic energy_norm)/d positions_A; fixed periodic shifts",
        "force_normalization": "target F_eV_A * force_length_A / Estd; prediction -d(sum E_norm)/dR_A * force_length_A",
        "virial_loss_active": settings["virial_weight"] > 0,
        "selection_objective": "E+F+W" if settings["virial_weight"] > 0 else "E+F only; W diagnostics excluded",
        "validation_only_scope": "full file parsed for source/split audit; only training and validation losses are computed",
        "legacy_reproduction_claim": False,
    })
    np.savez(output / "split.npz", **parts, dataset_sha256=dataset_hash)
    train_loader = _loader(graphs, parts["train_indices"], settings, training=True, generator=generator, pin_memory=pin_memory)
    val_loader = _loader(graphs, parts["val_indices"], settings, pin_memory=pin_memory)
    for epoch in range(first_epoch, settings["epochs"]):
        started = time.monotonic()
        model.train()
        observations, counts = [], []
        order = BatchOrderFingerprint()
        for batch in train_loader:
            order.update(batch)
            optimizer.zero_grad(set_to_none=True)
            losses, count = backward_logical_batch(model, batch, stats, settings, force_settings,
                                                   device, non_blocking=pin_memory)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            observations.append(torch.cat((losses, gradient_norm.detach().reshape(1))))
            counts.append(count)
        averaged_losses, observed_rows = _loss_averages(observations, counts)
        metrics = validation_loss(model, val_loader, stats, settings, device,
                                    force_settings=force_settings, non_blocking=pin_memory)
        val_loss = metrics["normalized_weighted_loss"]
        scheduler.step()
        improved = val_loss < best_loss - settings["min_delta"]
        if improved:
            best_loss, bad_epochs = val_loss, 0
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        state = {
            "format": FORMAT, "package_version": __version__, "epoch": epoch,
            "model_spec": spec, "model_state": model.state_dict(), "stats": stats,
            "data_spec": config["data"], "config": config, "dataset_sha256": dataset_hash,
            "force_settings": force_settings, "force_label_audit": force_audit,
            "split": {key: torch.from_numpy(value) for key, value in parts.items()},
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_loss, "bad_epochs": bad_epochs, "best_model_state": best_weights,
            "best_epoch": best_epoch,
            "initial_state_sha256": initial_hash, "execution_options": execution_options,
            "train_order": {"epoch": epoch, **order.record()},
            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(), "numpy_rng": _numpy_rng_state(),
        }
        if improved:
            atomic_save(state, output / "model_best.ckpt")
        stopping = bool(settings["patience"] and bad_epochs >= settings["patience"])
        if (epoch + 1) % checkpoint_interval == 0 or epoch + 1 == settings["epochs"] or stopping:
            atomic_save(state, output / "model_latest.ckpt")
        if epoch + 1 in milestones:
            milestone = output / f"model_epoch_{epoch + 1:06d}.ckpt"
            _save_immutable_checkpoint(state, milestone)
            ledger["checkpoints"][str(epoch + 1)] = {"path": milestone.name, "sha256": file_digest(milestone),
                                                   "epoch_zero_based": epoch}
            write_json(ledger_path, ledger)
        row = {
            "epoch": epoch, "train_loss": averaged_losses[0], "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started,
        }
        _append_csv(history, row)
        order_row = {"epoch": epoch, **order.record()}
        _append_csv(order_history, order_row)
        norms = [float(value) for value in observed_rows[:, 4]]
        _append_csv(diagnostics, {
            "epoch_zero_based": epoch, "completed_epochs": epoch + 1,
            "energy_weight": settings["energy_weight"], "virial_weight": settings["virial_weight"],
            "force_weight": force_settings["force_weight"], "force_length_A": force_settings["force_length_A"],
            "virial_loss_active": settings["virial_weight"] > 0,
            "seed": config["run"]["seed"],
            "train_E_unweighted_normalized_huber": averaged_losses[1],
            "train_W_unweighted_normalized_huber": averaged_losses[2],
            "train_F_unweighted_normalized_huber": averaged_losses[3],
            "validation_E_unweighted_normalized_huber": metrics["normalized_energy_loss"],
            "validation_W_unweighted_normalized_huber": metrics["normalized_virial_loss"],
            "validation_F_unweighted_normalized_huber": metrics["normalized_force_loss"],
            "val_weighted_normalized_huber": val_loss,
            "gradient_steps": len(norms), "preclip_gradient_norm_mean": sum(norms) / len(norms),
            "preclip_gradient_norm_max": max(norms),
            "gradient_clipped_fraction": sum(value > settings["gradient_clip"] for value in norms) / len(norms),
        })
        print(f"epoch={epoch} train={row['train_loss']:.6g} val={val_loss:.6g}", flush=True)
        if stopping:
            break
    # Write the true best model even when resuming into a different output directory.
    best_export = dict(state, model_state=best_weights, epoch=best_epoch)
    best_export.pop("optimizer_state")
    best_export.pop("scheduler_state")
    best_export["checkpoint_role"] = "best model for prediction; use latest for resume"
    atomic_save(best_export, output / "model_best.ckpt")
    metrics = dict(metrics, partition="val", checkpoint_epoch_zero_based=epoch)
    write_json(output / "validation_loss.json", metrics)
    write_json(output / "run_status.json", {"completed_epochs": epoch + 1, "requested_epochs": settings["epochs"],
                                          "best_epoch_zero_based": best_epoch, "best_val_loss": best_loss,
                                          "test_evaluated": False, "execution_options": execution_options})
    return metrics
