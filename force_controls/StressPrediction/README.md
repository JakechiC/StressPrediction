# StressPrediction source

Install this subproject in its own virtual environment using Python 3.10–3.12, with compatible PyTorch, PyG and torch-scatter builds already installed:

```bash
python -m pip install -e .
stress-prediction --help
stress-prediction check-config --config configs/painn_direct_ar.yaml
```

After providing your dataset and paths in a configuration, launch training explicitly:

```bash
stress-prediction train --config configs/painn_direct_ar.yaml
stress-prediction predict --checkpoint runs/painn_direct_ar/model_best.ckpt --input structures.xyz --output predictions.npz
```

Input files use ASE-readable structures with per-atom `energy_per_atom` and `stress_per_atom` arrays, and an optional `representative_mask`. Set the label units explicitly in the configuration. Internal energy and virial units are eV; virial is energy-valued, without atomic-volume division. Raw prediction writes per-atom energy/virial arrays, structure coordinates, cells, periodicity and identifiers.

Training saves checkpoints and training/validation losses. It does not run a final test-set analysis or produce plots. Checkpoint selection uses the same weighted, atom-averaged validation loss as the input source. The release changes reporting/API scope; it is not a byte-identical historical execution bundle. Use a new output directory for new runs; resume checkpoints produced by this version with its `--resume` option.

Graphene commands retain policy training and candidate generation:

```bash
stress-prediction graphene train --config configs/graphene_design.yaml
stress-prediction graphene evaluate --config configs/graphene_design.yaml
stress-prediction graphene random --config configs/graphene_design.yaml
```

Supply an explicit surrogate checkpoint. The `evaluate` command here generates candidate structures; it performs no AIREBO comparison. The reward uses full-matrix virial norms and requires an explicit threshold unit. Candidate files retain structures, raw tensors and generation metadata. The example uses `20 GPa*A^3`, as in the paper.

SOAP/FPS input preprocessing remains available through `stress-prediction select --help`; install optional dependencies with `python -m pip install -e ".[selection]"`.

For E+F(+W) training, use the dedicated API with a force configuration:

```python
from stress_prediction.force_training import load_config, train
config = load_config("force_config.yaml")
train(config)
```

The force configuration must include `force_weight`, `force_length_A`, `force_key`, and `force_unit` under `force`, and use PaiNNDerivative or TensorNetDerivative. The ordinary `train` CLI remains the E+W route. A force-field entry does not switch that CLI into force training.

Datasets, trained weights, paper figures and offline analysis scripts are not bundled. Example configurations require external data. The root MIT license and third-party notices apply to this subproject.
