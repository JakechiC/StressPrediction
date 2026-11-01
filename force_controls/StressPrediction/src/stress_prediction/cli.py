"""Explicit model training, prediction and input selection commands."""
import argparse
import json
from pathlib import Path
import sys

def parser():
    root = argparse.ArgumentParser(prog="stress-prediction", description="Per-atom energy and virial learning")
    root.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-config", help="Validate configuration without loading data or models")
    train = commands.add_parser("train", help="Train E+W models with validation-loss checkpoint selection")
    train.add_argument("--resume")
    train.add_argument("--milestone-epochs", type=int, nargs="*", default=[])
    train.add_argument("--checkpoint-interval", type=int, default=1)
    train.add_argument("--pin-memory", action="store_true")
    for command in (check, train):
        command.add_argument("--config", required=True)
        command.add_argument("--data")
        command.add_argument("--output-dir")
        command.add_argument("--split-file")
        command.add_argument("--device")
        command.add_argument("--seed", type=int)
        command.add_argument("--model")
    predict = commands.add_parser("predict", help="Write raw model predictions for unlabeled structures")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--allow-legacy-pickle", action="store_true")
    commands.add_parser("select", help="SOAP/FPS input preprocessing; select --help for arguments")
    commands.add_parser("graphene", help="Policy training and candidate generation; graphene --help for arguments")
    return root

def _overrides(args):
    result = {}
    for argument, (group, key) in {
        "data": ("data", "path"), "split_file": ("split", "path"), "device": ("run", "device"),
        "seed": ("run", "seed"), "model": ("model", "name"), "output_dir": ("run", "output_dir"),
    }.items():
        value = getattr(args, argument, None)
        if value is not None:
            result.setdefault(group, {})[key] = value
    return result

def predict_file(checkpoint, input_path, output, *, device="cpu", allow_legacy_pickle=False):
    import numpy as np
    from ase.io import iread
    from .checkpoints import load_predictor
    from .data import file_digest
    from .training import write_json

    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("Prediction output must use the .npz extension")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Prediction output or metadata already exists: {output}")
    predictor = load_predictor(checkpoint, device, allow_legacy_pickle)
    arrays = {key: [] for key in ("energy_pred_eV", "virial_pred_eV", "frame_index", "atom_index", "positions_A", "atomic_numbers")}
    cells, pbc = [], []
    for index, atoms in enumerate(iread(input_path, index=":")):
        energy, virial = predictor(atoms)
        values = (energy, virial, np.full(len(atoms), index, dtype=np.int64), np.arange(len(atoms)), atoms.positions, atoms.numbers)
        for key, value in zip(arrays, values):
            arrays[key].append(value)
        cells.append(atoms.cell.array)
        pbc.append(predictor.data_spec.get("pbc") if predictor.data_spec.get("pbc") is not None else atoms.pbc)
    if not cells:
        raise ValueError("Input structure file is empty")
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Prediction contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, cells_A=np.asarray(cells), pbc=np.asarray(pbc, dtype=bool))
    write_json(output.with_suffix(".json"), {
        "energy_unit": "eV", "virial_unit": "eV", "positions_unit": "angstrom",
        "tensor_storage": "N,3,3 symmetric", "n_frames": len(cells),
        "checkpoint_sha256": file_digest(checkpoint), "input_sha256": file_digest(input_path),
        "data_spec": predictor.data_spec, "selection": "all input atoms; no target labels required",
    })

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "graphene":
        from .rl.cli import main as graphene_main
        return graphene_main(argv[1:])
    if argv and argv[0] == "select":
        from .selection import main as selection_main
        return selection_main(argv[1:])
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command in ("check-config", "train"):
            from .config import load_config, model_spec
            config = load_config(args.config, _overrides(args))
            model_spec(config)
            if args.command == "check-config":
                print(json.dumps(config, indent=2, ensure_ascii=False))
            else:
                from .training import train
                train(config, resume=args.resume, milestone_epochs=args.milestone_epochs,
                      checkpoint_interval=args.checkpoint_interval, pin_memory=args.pin_memory)
        elif args.command == "predict":
            predict_file(args.checkpoint, args.input, args.output, device=args.device,
                         allow_legacy_pickle=args.allow_legacy_pickle)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, KeyError, ImportError) as error:
        root.exit(2, f"error: {error}\n")

if __name__ == "__main__":
    raise SystemExit(main())
