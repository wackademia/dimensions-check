"""HTTPS LAN server for phone-based GarmentIQ measurements."""

import argparse
import base64
import io
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import run_live
from garmentiq import LiveDetector


app = Flask(__name__, static_folder="phone_web")
inference_lock = threading.Lock()
agent = None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/measure")
def measure():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify(error="Missing camera frame"), 400
    pixels_per_inch = request.form.get("pixels_per_inch", type=float)
    class_name = request.form.get("class_name") or None
    if not pixels_per_inch or pixels_per_inch <= 0:
        return jsonify(error="Calibrate with the 12-inch steel ruler first"), 400

    encoded = np.frombuffer(uploaded.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Could not decode camera frame"), 400

    detector = LiveDetector(
        agent,
        pixels_per_unit=pixels_per_inch,
        unit_name="in",
        class_name=class_name,
    )
    with inference_lock:
        result = detector.process_frame(frame)
    ok, jpeg = cv2.imencode(".jpg", result.frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return jsonify(error="Could not encode result"), 500
    return jsonify(
        garment_class=result.garment_class,
        inference_seconds=round(result.inference_seconds, 2),
        measurements=result.measurements,
        image="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
    )


def main():
    global agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve plain HTTP (safe for camera access on localhost only)",
    )
    args = parser.parse_args()
    print("Loading GarmentIQ models. This can take about a minute on CPU...")
    agent = run_live.build_agent()
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
