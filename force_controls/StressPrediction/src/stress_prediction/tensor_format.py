"""Tensor storage and physical-unit conversion for input labels."""
import numpy as np
from ase import units
EV_TO_GPA_A3 = float(units.eV / units.GPa)

def to_matrix(values):
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        result = array
    elif array.ndim == 2 and array.shape[1] == 6:
        result = np.zeros((len(array), 3, 3), dtype=array.dtype)
        result[:, 0, 0], result[:, 1, 1], result[:, 2, 2] = array[:, :3].T
        for col, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
            result[:, i, j] = result[:, j, i] = array[:, col]
    else:
        raise ValueError(f"Expected (N,6) or (N,3,3) virials, got {array.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Virials contain non-finite values")
    if not np.allclose(result, result.transpose(0, 2, 1), rtol=1e-5, atol=1e-7):
        raise ValueError("This pipeline requires symmetric per-atom virials")
    return result

def virial_factor_to_eV(unit):
    if unit == "eV":
        return 1.0
    if unit in ("GPa*A^3", "GPa·Å³"):
        return 1.0 / EV_TO_GPA_A3
    if unit == "bar*A^3":
        return 1e-4 / EV_TO_GPA_A3
    raise ValueError(f"Unsupported energy-valued virial unit: {unit!r}")
