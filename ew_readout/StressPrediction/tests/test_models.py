"""Small synthetic graph checks, with optional CUDA reduction regressions.

Optional source compatibility checks load a trusted original source directory
specified with STRESS_PREDICTION_LEGACY_SOURCE. They construct random models;
they do not deserialize checkpoints or start training/evaluation jobs.
"""

import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

from stress_prediction.models import MODEL_NAMES, build_model
from stress_prediction.models import painn_utils
from stress_prediction.models.painn_utils import EdgeBasedStressOutput, EquivariantStressOutput


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_parameters(name):
    parameters = {"hidden_channels": 8, "num_layers": 2, "num_rbf": 6}
    if name.startswith("TensorNet"):
        parameters.update(cutoff_upper=5.0, max_z=10)
    else:
        parameters.update(cutoff=5.0, num_elements=10)
    return parameters


def tetrahedron(dtype=torch.float32):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.2, 1.3, 0.1], [0.1, 0.3, 1.6]],
        dtype=dtype,
    )
    pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return Batch.from_data_list(
        [
            Data(
                positions=positions,
                z=torch.tensor([6, 6, 6, 6]),
                edge_index=edge_index,
                shifts=torch.zeros(len(pairs), 3, dtype=dtype),
                num_nodes=4,
            )
        ]
    )


def disconnected_graphs(dtype):
    """A connected component plus a trailing isolated atom, and no-edge graph."""
    partial = tetrahedron(dtype).to_data_list()[0]
    partial.positions = torch.cat((partial.positions, torch.tensor([[30., 0., 0.]], dtype=dtype)))
    partial.z = torch.cat((partial.z, torch.tensor([6])))
    partial.num_nodes = 5
    edgeless = Data(
        positions=torch.tensor([[0., 0., 0.], [20., 0., 0.], [40., 0., 0.]], dtype=dtype),
        z=torch.tensor([6, 6, 6]), edge_index=torch.empty((2, 0), dtype=torch.long),
        shifts=torch.empty((0, 3), dtype=dtype), num_nodes=3,
    )
    return partial, edgeless


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("case", ["partial", "edgeless", "mixed"])
def test_isolated_and_edgeless_graphs_keep_all_outputs_and_finite_gradients(name, dtype, case):
    torch.manual_seed(61)
    model = build_model(name, small_parameters(name)).to(dtype=dtype)
    partial, edgeless = disconnected_graphs(dtype)
    inputs = {"partial": [partial], "edgeless": [edgeless], "mixed": [partial, edgeless]}[case]
    graph = Batch.from_data_list(inputs)
    model.eval()
    with torch.no_grad():
        energy, virial = model(graph)
    assert energy.shape == (graph.num_nodes, 1)
    assert virial.shape == (graph.num_nodes, 3, 3)
    assert energy.dtype == virial.dtype == dtype
    assert torch.isfinite(energy).all() and torch.isfinite(virial).all()
    torch.testing.assert_close(virial, virial.transpose(-1, -2))
    # E+W backward also exercises the derivative models' higher-order graph.
    model.train()
    energy, virial = model(graph)
    (energy.square().mean() + virial.square().mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isolated_nodes_and_mixed_batches_do_not_change_connected_component(name, dtype):
    torch.manual_seed(67)
    model = build_model(name, small_parameters(name)).to(dtype=dtype).eval()
    partial, edgeless = disconnected_graphs(dtype)
    with torch.no_grad():
        connected_result = model(tetrahedron(dtype))
        partial_result = model(Batch.from_data_list([partial]))
        edgeless_result = model(Batch.from_data_list([edgeless]))
        mixed_result = model(Batch.from_data_list([partial, edgeless]))
    tolerance = {"rtol": 1e-5, "atol": 1e-6} if dtype == torch.float32 else {"rtol": 1e-10, "atol": 1e-11}
    for connected, with_isolated, alone_empty, mixed in zip(
        connected_result, partial_result, edgeless_result, mixed_result
    ):
        torch.testing.assert_close(with_isolated[:4], connected, **tolerance)
        torch.testing.assert_close(mixed[:5], with_isolated, **tolerance)
        torch.testing.assert_close(mixed[5:], alone_empty, **tolerance)
    # Edge sums and TensorNet's zero tensor embedding imply zero virial for
    # no-neighbor atoms. Do not impose this on heads with a learned p(s) I term.
    if name in ("PaiNNEdge", "PaiNNDerivative", "TensorNetDirect", "TensorNetDerivative"):
        torch.testing.assert_close(partial_result[1][4], torch.zeros((3, 3), dtype=dtype), rtol=0, atol=0)
        torch.testing.assert_close(edgeless_result[1], torch.zeros((3, 3, 3), dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtype_and_symmetric_output(name, dtype):
    torch.manual_seed(17)
    model = build_model(name, small_parameters(name)).to(dtype=dtype).eval()
    # Derivative paths must re-enable autograd inside ordinary no_grad contexts.
    with torch.no_grad():
        energy, virial = model(tetrahedron(dtype))
    assert energy.shape == (4, 1)
    assert virial.shape == (4, 3, 3)
    assert energy.dtype == virial.dtype == dtype
    assert torch.isfinite(energy).all()
    assert torch.isfinite(virial).all()
    torch.testing.assert_close(virial, virial.transpose(-1, -2))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_rotation_equivariance(name):
    torch.manual_seed(23)
    model = build_model(name, small_parameters(name)).double().eval()
    graph = tetrahedron(torch.float64)
    rotation = torch.tensor(
        [[0.36, -0.8, 0.48], [0.48, 0.6, 0.64], [-0.8, 0.0, 0.6]],
        dtype=torch.float64,
    )
    rotated = graph.clone()
    rotated.positions = graph.positions @ rotation.T
    rotated.shifts = graph.shifts @ rotation.T
    energy, virial = model(graph)
    rotated_energy, rotated_virial = model(rotated)
    torch.testing.assert_close(rotated_energy, energy, rtol=1e-8, atol=1e-9)
    expected = rotation @ virial @ rotation.T
    torch.testing.assert_close(rotated_virial, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("name", ["PaiNNDerivative", "TensorNetDerivative"])
def test_derivative_virial_supports_parameter_backpropagation(name):
    torch.manual_seed(31)
    model = build_model(name, small_parameters(name)).double().train()
    _, virial = model(tetrahedron(torch.float64))
    # One analytical backward pass checks the higher-order graph, without an
    # optimizer, parameter update, data loader, or training loop.
    virial.square().sum().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


@pytest.mark.parametrize("name", ["TensorNetDirect", "TensorNetDerivative"])
def test_tensornet_constructor_dtype(name):
    parameters = small_parameters(name)
    parameters["dtype"] = torch.float64
    model = build_model(name, parameters).eval()
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    energy, virial = model(tetrahedron(torch.float64))
    assert energy.dtype == virial.dtype == torch.float64


def test_single_dyad_head_preserves_its_documented_restriction_and_batch_axis():
    torch.manual_seed(41)
    head = EquivariantStressOutput(8).double()
    scalars = torch.randn(1, 8, dtype=torch.float64)
    vectors = torch.randn(1, 3, 8, dtype=torch.float64)
    output = head(scalars, vectors)
    assert output.shape == (1, 3, 3)
    eigenvalues = torch.linalg.eigvalsh(output)
    torch.testing.assert_close(eigenvalues[:, 0], eigenvalues[:, 1])
    assert torch.all(eigenvalues[:, 2] >= eigenvalues[:, 1])


def test_edge_head_removes_only_antisymmetric_reduction_roundoff(monkeypatch):
    """Make the accumulation-order issue reproducible without a CUDA race."""
    torch.manual_seed(71)
    head = EdgeBasedStressOutput(4).double()
    features = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 1, 2]])
    directions = torch.tensor(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]],
        dtype=torch.float64, requires_grad=True,
    )
    source, target = edge_index
    coefficients = head.edge_mlp(torch.cat((features[source], features[target]), dim=1))
    contributions = coefficients.unsqueeze(2) * directions.unsqueeze(2) * directions.unsqueeze(1)
    original_scatter = painn_utils.scatter
    reference = original_scatter(contributions, target, 0, dim_size=4)
    skew_roundoff = torch.zeros_like(reference)
    skew_roundoff[:, 0, 1] = 2. ** -20
    skew_roundoff[:, 1, 0] = -(2. ** -20)

    def scatter_with_roundoff(*args, **kwargs):
        return original_scatter(*args, **kwargs) + skew_roundoff

    monkeypatch.setattr(painn_utils, "scatter", scatter_with_roundoff)
    actual = head(features, edge_index, directions)
    # The axis dyads have zero off-diagonal entries. Projection removes the
    # injected skew part exactly and leaves the original dyad sum unchanged.
    torch.testing.assert_close(actual, actual.transpose(-1, -2), rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    inputs = (features, directions, *head.parameters())
    actual_gradients = torch.autograd.grad(actual.sum(), inputs)
    reference_gradients = torch.autograd.grad(reference.sum(), inputs)
    for actual_gradient, reference_gradient in zip(actual_gradients, reference_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction regression")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_edge_head_cuda_cancelling_dyads_are_exactly_symmetric(dtype):
    """Exercise parallel scatter without assuming a particular atomic order."""
    torch.manual_seed(73)
    head = EdgeBasedStressOutput(8).to(device="cuda", dtype=dtype)
    features = torch.randn(64, 8, device="cuda", dtype=dtype, requires_grad=True)
    pair_ids = torch.arange(2048, device="cuda")
    edge_index = torch.stack((pair_ids % 64, pair_ids // 32)).repeat_interleave(2, dim=1)
    base_directions = torch.randn(2048, 3, device="cuda", dtype=dtype)
    base_directions = base_directions / base_directions.norm(dim=1, keepdim=True)
    mirrored_directions = base_directions.clone()
    mirrored_directions[:, 0] *= -1
    directions = torch.stack((base_directions, mirrored_directions), dim=1).reshape(-1, 3)
    directions.requires_grad_(True)
    actual = head(features, edge_index, directions)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, actual.transpose(-1, -2), rtol=0, atol=0)
    # Reference the same edge contributions, summed on CPU in float64. The
    # bound scales with the absolute contribution sum and float precision;
    # each target receives 64 edges, including exact cancelling xy/xz pairs.
    with torch.no_grad():
        source, target = edge_index
        coefficients = head.edge_mlp(torch.cat((features[source], features[target]), dim=1))
        dyads = directions.unsqueeze(2) * directions.unsqueeze(1)
        contributions = (coefficients.unsqueeze(2) * dyads).cpu().double()
        reference = torch.zeros(64, 3, 3, dtype=torch.float64)
        absolute_sum = torch.zeros_like(reference)
        reference.index_add_(0, target.cpu(), contributions)
        absolute_sum.index_add_(0, target.cpu(), contributions.abs())
        error_bound = 128 * torch.finfo(dtype).eps * absolute_sum.clamp_min(1)
        assert torch.all((actual.detach().cpu().double() - reference).abs() <= error_bound)
    actual.square().mean().backward()
    gradients = [features.grad, directions.grad, *(parameter.grad for parameter in head.parameters())]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_factory_rejects_unknown_names_and_parameters_without_mutating_input():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("painn")
    with pytest.raises(TypeError, match="Invalid parameters"):
        build_model("PaiNNDirect", {"checkpoint_path": "unused.ckpt"})
    with pytest.raises(TypeError, match="mapping"):
        build_model("PaiNNDirect", [1, 2])
    parameters = small_parameters("PaiNNDirect")
    before = parameters.copy()
    build_model("PaiNNDirect", parameters)
    assert parameters == before


@pytest.fixture(scope="module")
def legacy_modules():
    source = os.environ.get("STRESS_PREDICTION_LEGACY_SOURCE")
    if not source:
        pytest.skip(
            "Optional: set STRESS_PREDICTION_LEGACY_SOURCE to trusted original code"
        )
    directory = Path(source).expanduser().resolve()
    names = ["painn_utils", "tensornet_utils", "painn", "tensornet", "tensorpainn"]
    previous = {name: sys.modules.get(name) for name in names}
    previous_bytecode_policy = sys.dont_write_bytecode
    loaded = {}
    try:
        # Source compatibility checks must not create caches beside legacy code.
        sys.dont_write_bytecode = True
        for name in names:
            path = directory / f"{name}.py"
            if not path.is_file():
                pytest.fail(f"Original source is missing: {path}")
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_legacy_state_dictionary_and_synthetic_predictions(name, legacy_modules):
    module_name = (
        "tensorpainn"
        if name == "TensorPaiNN"
        else ("tensornet" if name.startswith("TensorNet") else "painn")
    )
    torch.manual_seed(47)
    original = getattr(legacy_modules[module_name], name)(
        **small_parameters(name)
    ).eval()
    packaged = build_model(name, small_parameters(name)).eval()
    original_state = original.state_dict()
    packaged_state = packaged.state_dict()
    assert original_state.keys() == packaged_state.keys()
    assert all(
        original_state[key].shape == packaged_state[key].shape for key in original_state
    )
    packaged.load_state_dict(original_state, strict=True)
    expected = original(tetrahedron())
    actual = packaged(tetrahedron())
    for original_tensor, packaged_tensor in zip(expected, actual):
        torch.testing.assert_close(packaged_tensor, original_tensor, rtol=0, atol=0)
