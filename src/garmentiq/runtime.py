"""Device selection and single-model GPU residency for memory-limited inference."""

import warnings

import torch


def select_device(requested="auto"):
    """Probe a CUDA kernel before choosing CUDA; explicit CUDA requests fail loudly."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return torch.device("cpu")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable (check the driver and PyTorch wheel)")
        device = torch.device("cuda:0")
        # Availability alone does not prove that the wheel supports this GPU.
        with torch.no_grad():
            x = torch.ones((1, 1, 8, 8), device=device)
            kernel = torch.ones((1, 1, 3, 3), device=device)
            torch.nn.functional.conv2d(x, kernel).sum().item()
        return device
    except RuntimeError as exc:
        message = f"CUDA check failed: {exc}. See docs/GTX1050TI_SETUP.md."
        if requested == "cuda":
            raise RuntimeError(message) from exc
        warnings.warn(message + " Using CPU.", RuntimeWarning)
        return torch.device("cpu")


class InferenceRuntime:
    """Serialized inference, FP32, with CPU retry after CUDA out-of-memory."""

    def __init__(self, device="auto", low_vram=None):
        self.device = select_device(device)
        self.low_vram = bool(low_vram)
        if low_vram is None and self.device.type == "cuda":
            self.low_vram = torch.cuda.get_device_properties(self.device).total_memory <= 4.5 * 1024**3
        self.models = []
        # Autotuning can use additional workspace; conservative default for 4 GB cards.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False

    def prepare(self, models):
        self.models = list(models)
        if not self.low_vram:
            try:
                for model in self.models:
                    model.to(self.device)
            except torch.cuda.OutOfMemoryError:
                self._fallback()

    def _fallback(self):
        warnings.warn("GPU memory exhausted; continuing on CPU. Restart to retry CUDA.", RuntimeWarning)
        self.device = torch.device("cpu")
        for model in self.models:
            model.to("cpu")
        torch.cuda.empty_cache()

    def run(self, function, model, **kwargs):
        try:
            model.to(self.device)
            return function(model=model, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda":
                raise
            # Retry after leaving the except block, releasing the failed forward's traceback.
            pass
        finally:
            if self.low_vram and self.device.type == "cuda":
                model.to("cpu")
                torch.cuda.empty_cache()
        self._fallback()
        return function(model=model, **kwargs)
