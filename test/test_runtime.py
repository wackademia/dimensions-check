"""Run with python -m unittest discover -s test -p test_runtime.py -v."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from garmentiq.runtime import InferenceRuntime, select_device
from garmentiq.classification import predict
from garmentiq.landmark.detection import load_model
from garmentiq.segmentation import extract


class TrackingModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


class RuntimeTests(unittest.TestCase):
    def test_cpu_does_not_probe_cuda(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA probe")):
            self.assertEqual(select_device("cpu").type, "cpu")

    def test_auto_falls_back_but_explicit_cuda_fails(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(select_device("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "CUDA check failed"):
                select_device("cuda")

    def gpu_runtime(self, memory_gb=4):
        with patch("garmentiq.runtime.select_device", return_value=torch.device("cuda:0")), \
             patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=memory_gb * 1024**3)):
            return InferenceRuntime()

    def test_four_gb_auto_offloads_after_each_stage(self):
        runtime = self.gpu_runtime()
        model = TrackingModel()
        runtime.prepare([model])
        self.assertEqual(model.moves, [])
        with patch("torch.cuda.empty_cache"):
            self.assertEqual(runtime.run(lambda model: 42, model), 42)
        self.assertEqual(model.moves, ["cuda:0", "cpu"])

    def test_larger_gpu_keeps_models_resident(self):
        runtime = self.gpu_runtime(8)
        model = TrackingModel()
        runtime.prepare([model])
        self.assertFalse(runtime.low_vram)
        self.assertEqual(model.moves, ["cuda:0"])

    def test_oom_retries_once_on_cpu_and_stays_on_cpu(self):
        runtime = self.gpu_runtime()
        model = TrackingModel()
        runtime.prepare([model])
        devices = []

        def forward(model):
            devices.append(str(runtime.device))
            if runtime.device.type == "cuda":
                raise torch.cuda.OutOfMemoryError("test allocation failure")
            return 7

        with patch("torch.cuda.empty_cache"), self.assertWarns(RuntimeWarning):
            self.assertEqual(runtime.run(forward, model), 7)
        self.assertEqual(devices, ["cuda:0", "cpu"])
        self.assertEqual(runtime.device.type, "cpu")
        self.assertEqual(model.moves[-1], "cpu")

    def test_other_errors_are_not_silenced_and_model_is_offloaded(self):
        runtime = self.gpu_runtime()
        model = TrackingModel()
        with patch("torch.cuda.empty_cache"), self.assertRaisesRegex(ValueError, "bad input"):
            runtime.run(lambda model: (_ for _ in ()).throw(ValueError("bad input")), model)
        self.assertEqual(model.moves[-1], "cpu")

    def test_landmark_checkpoint_prefix_and_cpu_placement(self):
        original = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pth"
            torch.save({"module." + k: v for k, v in original.state_dict().items()}, path)
            loaded = load_model(str(path), torch.nn.Linear(3, 2), device="cpu")
        self.assertNotIsInstance(loaded, torch.nn.DataParallel)
        self.assertTrue(torch.equal(loaded.weight, original.weight))
        self.assertFalse(loaded.training)

    def test_classification_follows_model_even_if_cuda_is_available(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 2)).eval()
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"):
            label, probabilities = predict(model, np.zeros((12, 12, 3), dtype=np.uint8), ["a", "b"], resize_dim=(8, 8))
        self.assertIn(label, ["a", "b"])
        self.assertAlmostEqual(sum(probabilities), 1, places=5)

    def test_small_segmentation_restores_original_pixel_grid(self):
        model = torch.nn.Conv2d(3, 1, 1).eval()
        original, mask = extract(model, np.zeros((240, 360, 3), dtype=np.uint8), resize_dim=(32, 32))
        self.assertEqual(original.shape, (240, 360, 3))
        self.assertEqual(mask.shape, (240, 360))


if __name__ == "__main__":
    unittest.main()
