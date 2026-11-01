"""Checkpoint-compatible model definitions and explicit constructor dispatch.

All models return ``(energy_per_atom, virial_per_atom)`` with shapes
``[N, 1]`` and ``[N, 3, 3]``. The models operate in the normalization space
chosen by training; a local virial has energy dimensions, not energy/volume.
"""

from collections.abc import Mapping
from inspect import signature
from typing import Any

from torch import nn

from .painn import PaiNNDerivative, PaiNNDirect, PaiNNEdge
from .painn_headonly import PaiNNHeadOnly
from .tensorpainn import TensorPaiNN
from .tensornet import TensorNetDerivative, TensorNetDirect

_MODEL_CLASSES = {
    cls.__name__: cls
    for cls in (
        PaiNNDirect,
        PaiNNDerivative,
        PaiNNEdge,
        PaiNNHeadOnly,
        TensorNetDirect,
        TensorNetDerivative,
        TensorPaiNN,
    )
}
MODEL_NAMES = tuple(_MODEL_CLASSES)


def build_model(name: str, parameters: Mapping[str, Any] | None = None) -> nn.Module:
    """Construct a model by its canonical, case-sensitive class name.

    ``parameters`` contains constructor keyword arguments only. No paths,
    training options, or unrecognized keys are silently consumed. PaiNN-family
    constructors use ``cutoff`` and ``num_elements``; TensorNet constructors
    use ``cutoff_upper`` and ``max_z``. For an existing checkpoint, retain the
    original architecture parameters and load its state dictionary strictly.

    Construction never loads data, reads checkpoints, or launches a job.
    """
    if not isinstance(name, str) or name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model {name!r}. Choose one of: {', '.join(MODEL_NAMES)}"
        )
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise TypeError("Model parameters must be a mapping of constructor keywords.")
    constructor = _MODEL_CLASSES[name]
    kwargs = dict(parameters)
    try:
        signature(constructor).bind(**kwargs)
    except TypeError as error:
        raise TypeError(f"Invalid parameters for {name}: {error}") from error
    return constructor(**kwargs)


__all__ = [
    "MODEL_NAMES",
    "build_model",
    "PaiNNDirect",
    "PaiNNDerivative",
    "PaiNNEdge",
    "PaiNNHeadOnly",
    "TensorNetDirect",
    "TensorNetDerivative",
    "TensorPaiNN",
]
