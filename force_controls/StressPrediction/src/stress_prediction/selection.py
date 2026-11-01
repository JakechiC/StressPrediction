"""Optional SOAP/FPS configuration selection; computation starts only on request.

The preprocessing follows the project's soap_select_fps.py (Ting Liang):
structure-averaged SOAP, optional row L2 normalization, optional feature
standardization, then sample FPS. Paths are provided by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np


def _load_backends():
    try:
        from ase.io import read, write
        from dscribe.descriptors import SOAP
        from sklearn.preprocessing import StandardScaler
        from skmatter.sample_selection import FPS
    except ImportError as exc:
        raise ImportError(
            "SOAP/FPS selection needs ASE, DScribe, scikit-learn and "
            "scikit-matter. Install the project's selection extra."
        ) from exc
    return read, write, SOAP, StandardScaler, FPS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    result = {}
    for package in ("ase", "dscribe", "scikit-learn", "scikit-matter", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not recorded"
    return result


def select_frames(
    input_path: str | Path,
    output_path: str | Path,
    n_select: int = 4000,
    *,
    species: Sequence[str] | None = None,
    r_cut: float = 5.0,
    n_max: int = 6,
    l_max: int = 4,
    sigma: float = 1.0,
    average: str = "inner",
    normalize: bool = True,
    standardize: bool = True,
    initial_index: int = 0,
    progress: bool = False,
    overwrite: bool = False,
) -> np.ndarray:
    """Select frames and save extxyz, ordered pool indices, and provenance.

    The output ``selected.extxyz`` is accompanied by ``selected.indices.npy``
    and ``selected.selection.json``. This is a real descriptor calculation;
    it is not invoked by import, installation, or model training.
    """
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input extended XYZ file does not exist: {source}")
    if source == destination:
        raise ValueError("Selection output must differ from the input pool.")
    if isinstance(n_select, bool) or not isinstance(n_select, int) or n_select < 1:
        raise ValueError("n_select must be a positive integer.")
    if n_max < 1 or l_max < 0 or r_cut <= 0 or sigma <= 0:
        raise ValueError("SOAP requires n_max>=1, l_max>=0, r_cut>0 and sigma>0.")
    if not np.isfinite([r_cut, sigma]).all():
        raise ValueError("SOAP cutoff and sigma must be finite.")
    if average not in {"inner", "outer"}:
        raise ValueError("Frame selection requires average='inner' or 'outer'.")

    index_path = destination.with_name(destination.stem + ".indices.npy")
    metadata_path = destination.with_name(destination.stem + ".selection.json")
    targets = (destination, index_path, metadata_path)
    if source in targets or len(set(targets)) != len(targets):
        raise ValueError("Output and sidecar paths must be distinct from the input.")
    for path in targets:
        if path.exists() and (not overwrite or not path.is_file()):
            raise FileExistsError(f"Refusing to overwrite output: {path}")

    read, write, SOAP, StandardScaler, FPS = _load_backends()
    frames = read(str(source), index=":", format="extxyz")
    if not frames:
        raise ValueError("The input pool contains no frames.")
    if n_select > len(frames):
        raise ValueError(f"Requested {n_select} frames from a pool of {len(frames)}.")
    if isinstance(initial_index, bool) or not isinstance(initial_index, int):
        raise ValueError("initial_index must be an integer pool index.")
    if not 0 <= initial_index < len(frames):
        raise ValueError("initial_index is outside the input pool.")

    observed_species = sorted({symbol for frame in frames for symbol in frame.get_chemical_symbols()})
    chosen_species = list(species) if species is not None else observed_species
    if not chosen_species or len(set(chosen_species)) != len(chosen_species):
        raise ValueError("species must contain unique element symbols.")
    if not set(observed_species).issubset(chosen_species):
        raise ValueError("Configured species omit one or more input elements.")

    soap_parameters = {
        "species": chosen_species,
        "periodic": True,
        "r_cut": r_cut,
        "n_max": n_max,
        "l_max": l_max,
        "sigma": sigma,
        "rbf": "gto",
        "average": average,
        "dtype": "float32",
    }
    soap = SOAP(**soap_parameters)
    descriptors = np.asarray([soap.create(frame) for frame in frames], dtype=np.float32)
    if descriptors.ndim != 2 or not np.isfinite(descriptors).all():
        raise ValueError("SOAP must produce one finite descriptor vector per frame.")
    if normalize:
        lengths = np.linalg.norm(descriptors, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        descriptors = descriptors / lengths
    if standardize:
        descriptors = StandardScaler().fit_transform(descriptors)
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptor preprocessing produced non-finite values.")

    if n_select == 1:
        selected = np.asarray([initial_index], dtype=np.int64)
    else:
        selector = FPS(
            n_to_select=n_select,
            initialize=initial_index,
            random_state=0,
            progress_bar=progress,
            full=False,
        )
        selector.fit(descriptors)
        selected = np.asarray(selector.selected_idx_, dtype=np.int64)
    if (
        selected.shape != (n_select,)
        or len(np.unique(selected)) != n_select
        or np.any(selected < 0)
        or np.any(selected >= len(frames))
    ):
        raise ValueError("FPS did not return the requested number of unique valid indices.")

    metadata = {
        "schema_version": 1,
        "input_filename": source.name,
        "input_sha256": _sha256(source),
        "pool_frames": len(frames),
        "selected_frames": n_select,
        "output_filename": destination.name,
        "indices_filename": index_path.name,
        "index_convention": "zero-based indices in the ordered input pool, in selection order",
        "soap": soap_parameters,
        "preprocessing": {"row_l2_normalization": normalize, "feature_standardization": standardize},
        "fps": {"initial_index": initial_index, "random_state": 0, "full": False},
        "versions": _versions(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage all three complete artifacts before replacing any destination.
    with tempfile.TemporaryDirectory(prefix=".selection-", dir=destination.parent) as folder:
        staged = [Path(folder) / path.name for path in targets]
        write(str(staged[0]), [frames[int(i)] for i in selected], format="extxyz")
        with staged[1].open("wb") as stream:
            np.save(stream, selected, allow_pickle=False)
        staged[2].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for temporary, target in zip(staged, targets):
            if target.exists() and not overwrite:
                raise FileExistsError(f"Output appeared during selection: {target}")
            temporary.replace(target)
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-select", type=int, default=4000)
    parser.add_argument("--species", nargs="+")
    parser.add_argument("--r-cut", type=float, default=5.0)
    parser.add_argument("--n-max", type=int, default=6)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--average", choices=("inner", "outer"), default="inner")
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = select_frames(
            args.input, args.output, args.n_select, species=args.species,
            r_cut=args.r_cut, n_max=args.n_max, l_max=args.l_max,
            sigma=args.sigma, average=args.average,
            normalize=not args.no_normalize, standardize=not args.no_standardize,
            initial_index=args.initial_index, progress=args.progress,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Selected {len(selected)} frames; wrote {args.output} and provenance sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
