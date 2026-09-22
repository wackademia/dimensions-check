"""HTTPS LAN server for phone-based GarmentIQ measurements."""

import argparse
import base64
import math
import mimetypes
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

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
snapshot_dir = None


def prompt_snapshot_dir():
    """Ask where to save inspection snapshots. Run interactively at startup since the
    server is typically launched fresh on each machine it runs on."""
    default_dir = Path("snapshots")
    answer = input(f"Where should inspection snapshots be saved? [{default_dir}]: ").strip()
    directory = Path(answer) if answer else default_dir
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Saving inspection snapshots to: {directory.resolve()}")
    return directory


def _table_fonts():
    try:
        return ImageFont.truetype("arial.ttf", 20), ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
        return font, font


def render_dimensions_table(measurements, decision, garment_class):
    font, header_font = _table_fonts()
    decision = decision or {}
    matched_reference = decision.get("reference")
    # Uncertain results still get a comparison column (the closest size, even though it
    # didn't produce exactly one waist match) instead of a blank "—" with no data to review.
    reference = matched_reference or decision.get("closest_reference") or {}
    if matched_reference:
        reference_column = "Reference"
    elif decision.get("closest_size"):
        reference_column = f'Closest: {decision["closest_size"]} (unmatched)'
    else:
        reference_column = "Reference"
    columns = ["Measurement", "Measured", reference_column, "Delta"]
    rows = []
    for name in sorted(measurements):
        value = measurements[name]
        ref = reference.get(name)
        rows.append([
            name,
            f'{value["distance"]:.2f} {value["unit"]}',
            f"{ref:.2f} in" if ref is not None else "—",
            f'{value["distance"] - ref:+.2f} in' if ref is not None else "—",
        ])
    size_text = decision.get("size") if decision else None
    header_line = f'{garment_class} · Size: {size_text or "uncertain"}'
    reason_line = decision.get("reason", "") if decision else ""

    scratch_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    padding = 12
    col_widths = [
        max(scratch_draw.textlength(str(line[i]), font=header_font if line is columns else font)
            for line in [columns] + rows) + padding * 2
        for i in range(len(columns))
    ]
    row_height = 30
    table_width = int(sum(col_widths))
    header_height = 66
    table_height = header_height + row_height * (len(rows) + 1)

    table = Image.new("RGB", (table_width, table_height), "white")
    draw = ImageDraw.Draw(table)
    draw.text((padding, 8), header_line, font=header_font, fill="black")
    draw.text((padding, 34), reason_line, font=font, fill=(80, 80, 80))
    x = 0
    for i, column in enumerate(columns):
        draw.text((x + padding, header_height + 6), column, font=header_font, fill="black")
        x += col_widths[i]
    draw.line((0, header_height + row_height, table_width, header_height + row_height), fill="black", width=2)
    for r, row in enumerate(rows):
        y = header_height + row_height * (r + 1)
        x = 0
        for i, cell in enumerate(row):
            draw.text((x + padding, y + 6), str(cell), font=font, fill="black")
            x += col_widths[i]
        draw.line((0, y + row_height, table_width, y + row_height), fill=(220, 220, 220), width=1)
    return table


def save_inspection_snapshot(annotated_bgr, measurements, decision, garment_class):
    """Compose the annotated photo with a dimensions table below it and save to disk."""
    photo = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    table = render_dimensions_table(measurements, decision, garment_class)
    width = max(photo.width, table.width)
    canvas = Image.new("RGB", (width, photo.height + table.height), "white")
    canvas.paste(photo, ((width - photo.width) // 2, 0))
    canvas.paste(table, ((width - table.width) // 2, photo.height))
    size_label = (decision.get("size") if decision else None) or "uncertain"
    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{garment_class}_{size_label}.jpg"
    path = snapshot_dir / filename
    canvas.save(path, quality=92)
    return path


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
    if underwear and measurements and snapshot_dir:
        try:
            saved_path = save_inspection_snapshot(annotated, measurements, decision, label)
            app.logger.info("Saved inspection snapshot to %s", saved_path)
        except Exception:
            app.logger.exception("Failed to save inspection snapshot")
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
    global agent, snapshot_dir
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
    snapshot_dir = prompt_snapshot_dir()
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
