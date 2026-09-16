"""Check CUDA and optional full static/phone inference before starting a server."""

import argparse
import io
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run both measurement endpoints with real models")
    parser.add_argument("--segmentation-size", type=int, choices=(256, 512, 768, 1024), default=512)
    args = parser.parse_args()
    import torch
    import torchvision
    from garmentiq.runtime import InferenceRuntime
    from download_models import ensure_models

    print(f"Python: {sys.version.split()[0]} ({sys.executable})", flush=True)
    print(f"PyTorch: {torch.__version__}; torchvision: {torchvision.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    runtime = InferenceRuntime("cuda" if args.require_cuda else "auto")
    if runtime.device.type == "cuda":
        info = torch.cuda.get_device_properties(runtime.device)
        print(f"GPU: {info.name}; {info.total_memory / 1024**3:.1f} GiB; capability {info.major}.{info.minor}")
        print(f"Wheel architectures: {torch.cuda.get_arch_list()}")
    # Exercise the compiled torchvision extension too (mismatched wheels fail here).
    boxes = torch.tensor([[0., 0., 10., 10.]], device=runtime.device)
    torchvision.ops.nms(boxes, torch.ones(1, device=runtime.device), 0.5)
    print(f"Device test OK: {runtime.device}; automatic low-VRAM mode: {runtime.low_vram}", flush=True)
    ensure_models(check_only=True)
    if args.smoke:
        import run_live
        import static_image_server
        import phone_server
        agent = run_live.build_agent("cuda" if args.require_cuda else "cpu", True, args.segmentation_size)
        static_image_server.agent = phone_server.agent = agent
        label, _ = agent.classify("demo_images/cloth_3.jpg")
        print(f"Classifier forward passed: {label}", flush=True)
        sample = (ROOT / "demo_images" / "cloth_3.jpg").read_bytes()
        for module, field in ((static_image_server, "image"), (phone_server, "frame")):
            response = module.app.test_client().post("/measure", data={
                field: (io.BytesIO(sample), "sample.jpg"), "pixels_per_inch": "100",
                "class_name": "vest dress",
            })
            payload = response.get_json()
            if response.status_code != 200:
                raise RuntimeError(f"{module.__name__}: {payload}")
            if not payload.get("measurements") or not all(
                math.isfinite(value["distance"]) and value["unit"] == "in"
                for value in payload["measurements"].values()
            ):
                raise RuntimeError("Invalid measurement result")
            print(f"{module.__name__}: HTTP 200, {len(payload['measurements'])} finite inch measurements", flush=True)
        if args.require_cuda and agent.runtime.device.type != "cuda":
            raise RuntimeError("Inference fell back to CPU; retry with --segmentation-size 256")
        print("Full pipeline smoke test passed (test scale is arbitrary, not an accuracy test).", flush=True)


if __name__ == "__main__":
    main()
