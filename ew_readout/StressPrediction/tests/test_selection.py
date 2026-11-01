"""No SOAP calculation: test selection orchestration with explicit fake backends."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from stress_prediction import selection


class FakeFrame:
    def __init__(self, index, descriptor):
        self.index = index
        self.descriptor = np.asarray(descriptor, dtype=np.float32)

    def get_chemical_symbols(self):
        return ["Ar"]


def fake_backend(*, indices=(0, 2)):
    frames = [FakeFrame(0, [3, 4]), FakeFrame(1, [0, 0]), FakeFrame(2, [4, 3])]
    soap = Mock()
    soap.create.side_effect = lambda frame: frame.descriptor
    soap_factory = Mock(return_value=soap)
    scaler = Mock()
    scaler.fit_transform.side_effect = lambda array: array - array.mean(axis=0)
    scaler_factory = Mock(return_value=scaler)
    fps = Mock(selected_idx_=np.asarray(indices))
    fps_factory = Mock(return_value=fps)
    reader = Mock(return_value=frames)

    def write(path, chosen, *, format):
        assert format == "extxyz"
        Path(path).write_text(",".join(str(frame.index) for frame in chosen), encoding="utf-8")

    backend = (reader, write, soap_factory, scaler_factory, fps_factory)
    return backend, soap_factory, scaler, fps_factory, fps


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name)
        self.source = self.directory / "pool.extxyz"
        self.source.write_text("fake pool, no scientific labels", encoding="utf-8")
        self.output = self.directory / "elsewhere" / "selected.extxyz"
        self.versions_patch = patch.object(selection, "_versions", return_value={"backend": "fake"})
        self.versions_patch.start()
        self.addCleanup(self.versions_patch.stop)

    def test_preserves_selection_order_and_preprocessing_metadata(self):
        backend, soap_factory, scaler, fps_factory, fps = fake_backend(indices=(2, 0))
        with patch.object(selection, "_load_backends", return_value=backend):
            actual = selection.select_frames(self.source, self.output, 2, initial_index=2)

        np.testing.assert_array_equal(actual, [2, 0])
        np.testing.assert_array_equal(np.load(self.output.with_suffix(".indices.npy"), allow_pickle=False), [2, 0])
        self.assertEqual(self.output.read_text(encoding="utf-8"), "2,0")
        np.testing.assert_allclose(scaler.fit_transform.call_args.args[0], [[0.6, 0.8], [0, 0], [0.8, 0.6]])
        self.assertEqual(soap_factory.call_args.kwargs["species"], ["Ar"])
        self.assertTrue(soap_factory.call_args.kwargs["periodic"])
        self.assertEqual(fps_factory.call_args.kwargs["initialize"], 2)
        self.assertEqual(fps.fit.call_args.args[0].shape, (3, 2))
        metadata = json.loads(self.output.with_suffix(".selection.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["input_filename"], "pool.extxyz")
        self.assertEqual(len(metadata["input_sha256"]), 64)
        self.assertEqual(metadata["selected_frames"], 2)
        self.assertTrue(metadata["preprocessing"]["feature_standardization"])
        self.assertNotIn(str(self.directory), json.dumps(metadata))

    def test_rejects_existing_sidecar_before_loading_backends(self):
        self.output.parent.mkdir()
        self.output.with_suffix(".indices.npy").write_bytes(b"existing")
        loader = Mock(side_effect=AssertionError("must not start selection"))
        with patch.object(selection, "_load_backends", loader):
            with self.assertRaises(FileExistsError):
                selection.select_frames(self.source, self.output, 2)
        loader.assert_not_called()
        self.assertEqual(self.output.with_suffix(".indices.npy").read_bytes(), b"existing")

    def test_rejects_invalid_fps_outputs_without_publishing(self):
        for indices in ((0, 0), (-1, 2), (0,), (0, 3)):
            with self.subTest(indices=indices):
                backend, *_ = fake_backend(indices=indices)
                with patch.object(selection, "_load_backends", return_value=backend):
                    with self.assertRaisesRegex(ValueError, "unique valid indices"):
                        selection.select_frames(self.source, self.output, 2)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.output.with_suffix(".indices.npy").exists())

    def test_refuses_input_overwrite_even_with_explicit_overwrite(self):
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "differ"):
            selection.select_frames(self.source, self.source, 1, overwrite=True)
        self.assertEqual(self.source.read_bytes(), before)

    def test_requested_count_cannot_exceed_pool(self):
        backend, soap_factory, *_ = fake_backend()
        with patch.object(selection, "_load_backends", return_value=backend):
            with self.assertRaisesRegex(ValueError, "pool of 3"):
                selection.select_frames(self.source, self.output, 4)
        soap_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
