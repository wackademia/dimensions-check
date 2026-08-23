"""Web demo for GarmentIQ's original static-image measurement pipeline."""

import argparse
import base64
import copy
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import run_live
from garmentiq import utils


app = Flask(__name__, static_folder="static_image_web")
inference_lock = threading.Lock()
agent = None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


def run_original_pipeline(image_rgb, forced_class=None):
    """Run the same stages used by ``tailor.measure`` on one in-memory image."""
    label = forced_class or agent.classify(image_rgb, verbose=False)[0]
    segmented = agent.segment(image_rgb)
    mask = segmented[1]
    detection_image = segmented[2] if len(segmented) == 3 else image_rgb
    coords, confidences, detection = agent.detect(label, detection_image)

    if agent.do_refine:
        coords, detection = agent.refine(
            class_name=label,
            detection_np=coords,
            detection_conf=confidences,
            detection_dict=detection,
            mask=mask,
        )
    if agent.do_derive:
        _, detection = agent.derive(
            class_name=label,
            detection_dict=detection,
            derivation_dict=agent.derivation_dict,
            landmark_coords=coords,
            np_mask=mask,
        )

    cleaned = utils.clean_detection_dict(label, "uploaded_image", detection)
    return label, cleaned["uploaded_image"]


def annotate(frame_bgr, garment, pixels_per_cm=None):
    frame = frame_bgr.copy()
    landmarks = garment["landmarks"]
    measurements = copy.deepcopy(garment["measurements"])
    for landmark in landmarks.values():
        cv2.circle(
            frame,
            (round(landmark["x"]), round(landmark["y"])),
            6,
            (55, 235, 100),
            -1,
        )

    y = 32
    for name, measurement in measurements.items():
        start = landmarks[measurement["landmarks"]["start"]]
        end = landmarks[measurement["landmarks"]["end"]]
        p1 = (round(start["x"]), round(start["y"]))
        p2 = (round(end["x"]), round(end["y"]))
        cv2.line(frame, p1, p2, (30, 220, 255), 3)
        if pixels_per_cm:
            measurement["distance"] /= pixels_per_cm
            measurement["unit"] = "cm"
        else:
            measurement["unit"] = "px"
        text = f'{name}: {measurement["distance"]:.2f} {measurement["unit"]}'
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 220, 255), 2)
        y += 30
    return frame, measurements


@app.post("/measure")
def measure():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify(error="Choose a garment image first"), 400
    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify(error="The uploaded file is not a readable image"), 400
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    forced_class = request.form.get("class_name") or None
    pixels_per_cm = request.form.get("pixels_per_cm", type=float)
    if pixels_per_cm is not None and pixels_per_cm <= 0:
        return jsonify(error="Pixels per cm must be greater than zero"), 400

    try:
        with inference_lock:
            label, garment = run_original_pipeline(image_rgb, forced_class)
        annotated, measurements = annotate(frame_bgr, garment, pixels_per_cm)
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Could not encode the annotated image")
        return jsonify(
            garment_class=label,
            measurements=measurements,
            image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            stages=["classification", "segmentation", "landmark detection", "refinement", "derivation", "distance calculation"],
        )
    except Exception as exc:
        return jsonify(error=f"Measurement failed: {exc}"), 500


def main():
    global agent
    parser = argparse.ArgumentParser(description="GarmentIQ static-image upload demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    print("Loading models for the original static-image pipeline...")
    agent = run_live.build_agent()
    print(f"Static demo ready: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
