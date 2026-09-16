"""Launch GarmentIQ live webcam measurement.

Press Q or Escape in the camera window to stop.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.derivation.derivation_dict import derivation_dict
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.segmentation.model_definition.birefnet import (
    BiRefNet,
    load_birefnet_config,
)
from garmentiq.runtime import InferenceRuntime


def add_runtime_arguments(parser):
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--low-vram", action="store_true", default=None,
                        help="Keep only the active model on GPU; automatic for GPUs with <=4.5 GB")
    parser.add_argument("--segmentation-size", type=int, choices=(256, 512, 768, 1024), default=None,
                        help="Internal mask resolution; defaults to 512 in low-VRAM mode, otherwise 1024")


def build_agent(device="auto", low_vram=None, segmentation_size=None):
    required = ["tiny_vit_inditex_finetuned.pt", "hrnet.pth", "birefnet/model.safetensors"]
    missing = [name for name in required if not (ROOT / "models" / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing models: " + ", ".join(missing) +
                                ". Run .venv\\Scripts\\python.exe scripts/download_models.py")
    runtime = InferenceRuntime(device, low_vram)
    size = segmentation_size if segmentation_size is not None else (512 if runtime.low_vram else 1024)
    if size not in (256, 512, 768, 1024):
        raise ValueError("segmentation_size must be 256, 512, 768, or 1024")
    print(f"Inference: {runtime.device}; low VRAM: {runtime.low_vram}; segmentation: {size}px; FP32")
    return giq.tailor(
        runtime=runtime,
        input_dir=str(ROOT),
        model_dir=str(ROOT / "models"),
        output_dir=str(ROOT / "live_output"),
        class_dict=garment_classes,
        do_derive=True,
        derivation_dict=derivation_dict,
        do_refine=True,
        classification_model_path="tiny_vit_inditex_finetuned.pt",
        classification_model_class=tinyViT,
        classification_model_args={
            "num_classes": len(garment_classes),
            "img_size": (120, 184),
            "patch_size": 6,
            "resize_dim": (120, 184),
            "normalize_mean": [0.8047, 0.7808, 0.7769],
            "normalize_std": [0.2957, 0.3077, 0.3081],
        },
        segmentation_model_path="birefnet/model.safetensors",
        segmentation_model_class=BiRefNet,
        segmentation_model_args={
            "model_config": load_birefnet_config(),
            "resize_dim": (size, size),
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "background_color": [102, 255, 102],
        },
        landmark_detection_model_path="hrnet.pth",
        landmark_detection_model_class=PoseHighResolutionNet(),
        landmark_detection_model_args={
            "scale_std": 200.0,
            "resize_dim": [288, 384],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
    )


def main():
    parser = argparse.ArgumentParser(description="GarmentIQ live webcam measurement")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--every", type=int, default=10, help="Process every Nth frame")
    parser.add_argument("--pixels-per-unit", type=float, default=None)
    parser.add_argument("--unit", default="cm")
    parser.add_argument("--class-name", choices=sorted(garment_classes), default=None)
    add_runtime_arguments(parser)
    args = parser.parse_args()

    print("Loading models; this can take several minutes on CPU...")
    detector = giq.LiveDetector(
        build_agent(args.device, args.low_vram, args.segmentation_size),
        pixels_per_unit=args.pixels_per_unit,
        unit_name=args.unit,
        class_name=args.class_name,
    )
    print("Opening camera. Press Q or Escape to stop.")
    detector.run(source=args.camera, process_every_n_frames=args.every)


if __name__ == "__main__":
    main()
