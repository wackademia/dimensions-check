"""HTTPS LAN server for phone-based GarmentIQ measurements."""

import argparse
import base64
import math
import mimetypes
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import run_live
from garmentiq import LiveDetector
from garmentiq.garment_classes import garment_classes
from garmentiq.underwear_sizes import (
    DEFAULT_TOLERANCE_IN, REFERENCES, InspectionRejected, estimate_size, uncertain,
)


# Windows registry MIME mappings may otherwise serve modules as text/plain.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
app = Flask(__name__, static_folder="phone_web")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
inference_lock = threading.Lock()
agent = None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/size-references")
def size_references():
    return jsonify(references=REFERENCES, tolerance_in=DEFAULT_TOLERANCE_IN, unit="in")


@app.errorhandler(413)
def frame_too_large(error):
    return jsonify(error="Frame upload exceeds 16 MB; reduce camera resolution."), 413


@app.post("/measure")
def measure():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify(error="Missing camera frame"), 400
    pixels_per_inch = request.form.get("pixels_per_inch", type=float)
    class_name = request.form.get("class_name") or None
    underwear = request.form.get("mode") == "underwear"
    if request.form.get("mode") not in (None, "", "measure", "underwear"):
        return jsonify(error="Unknown inspection mode"), 400
    if not pixels_per_inch or not math.isfinite(pixels_per_inch) or pixels_per_inch <= 0:
        return jsonify(error="Calibrate with the 12-inch steel ruler first"), 400
    if class_name is not None and class_name not in garment_classes:
        return jsonify(error="Unknown garment type"), 400
    if underwear:
        class_name = "shorts"

    encoded = np.frombuffer(uploaded.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if frame is None:
        return jsonify(error="Could not decode camera frame"), 400
    if min(frame.shape[:2]) < 64 or frame.shape[0] * frame.shape[1] > 12_000_000:
        return jsonify(error="Use a frame between 64 pixels per side and 12 megapixels."), 400
    if agent is None:
        return jsonify(error="Models are not ready; wait for server startup."), 503

    detector = LiveDetector(
        agent,
        pixels_per_unit=pixels_per_inch,
        unit_name="in",
        class_name=class_name,
        inspect_underwear=underwear,
    )
    # Never accumulate stale conveyor frames behind another camera's request.
    if not inference_lock.acquire(blocking=False):
        return jsonify(error="The model is busy. Wait and inspect this item again."), 429
    started = time.perf_counter()
    try:
        try:
            result = detector.process_frame(frame)
            measurements = result.measurements
            decision = estimate_size(measurements) if underwear else None
            annotated = result.frame
            label = result.garment_class
        except InspectionRejected as exc:
            measurements, annotated, label = {}, frame, "shorts"
            decision = uncertain(str(exc))
    except Exception:
        app.logger.exception("Camera inference failed")
        return jsonify(error="Inference failed. Check the server log and inspect the item again."), 500
    finally:
        inference_lock.release()
    ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return jsonify(error="Could not encode result"), 500
    return jsonify(
        garment_class=label,
        inference_seconds=round(time.perf_counter() - started, 2),
        measurements=measurements,
        size_estimate=decision,
        image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
    )


def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    run_live.add_runtime_arguments(parser)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve plain HTTP (safe for camera access on localhost only)",
    )
    args = parser.parse_args()
    print("Loading GarmentIQ models. This can take about a minute on CPU...")
    agent = run_live.build_agent(args.device, args.low_vram, args.segmentation_size)
    scheme = "HTTP" if args.http else "HTTPS"
    print(f"Phone server ready on {scheme} port {args.port}")
    app.run(
        host=args.host,
        port=args.port,
        ssl_context=None if args.http else "adhoc",
        threaded=True,
    )


if __name__ == "__main__":
    main()
