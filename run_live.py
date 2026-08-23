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


def build_agent():
    return giq.tailor(
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
            "resize_dim": (1024, 1024),
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
    args = parser.parse_args()

    print("Loading models; this can take several minutes on CPU...")
    detector = giq.LiveDetector(
        build_agent(),
        pixels_per_unit=args.pixels_per_unit,
        unit_name=args.unit,
        class_name=args.class_name,
    )
    print("Opening camera. Press Q or Escape to stop.")
    detector.run(source=args.camera, process_every_n_frames=args.every)


if __name__ == "__main__":
    main()
