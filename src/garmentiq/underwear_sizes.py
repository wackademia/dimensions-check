"""Conservative flat-garment reference matching, not a trained size classifier."""

import math

import cv2
import numpy as np


# Measured columns of the four reference screenshots supplied by the user.
# These are flat widths/lengths in inches, NOT body circumferences or industry sizes.
REFERENCES = {
    "M": {"front rise": 11.24, "full length": 11.36, "hips": 14.62, "waist": 13.69},
    "L": {"front rise": 11.14, "full length": 12.65, "hips": 16.72, "waist": 14.62},
    "XL": {"front rise": 11.30, "full length": 12.16, "hips": 17.05, "waist": 15.79},
    "2XL": {"front rise": 12.28, "full length": 12.39, "hips": 18.09, "waist": 16.84},
}
DEFAULT_TOLERANCE_IN = 0.50  # Provisional measurement allowance, not a factory specification.


class InspectionRejected(ValueError):
    """The frame is unsuitable for a size decision."""


def uncertain(reason):
    return {"status": "uncertain", "size": None, "reason": reason}


def estimate_size(measurements, tolerance_in=DEFAULT_TOLERANCE_IN):
    """Accept exactly one reference within tolerance on ALL four dimensions.

    A nearest neighbour alone is not sufficient: unknown sizes must be rejected.
    RMSE is reported as a distance in inches, never as a probability/confidence.
    """
    if not math.isfinite(tolerance_in) or tolerance_in <= 0:
        raise ValueError("Size tolerance must be finite and positive")
    values = {}
    for name in REFERENCES["M"]:
        value = measurements.get(name, {})
        distance = value.get("distance")
        if (value.get("unit") != "in" or isinstance(distance, bool)
                or not isinstance(distance, (int, float))
                or not math.isfinite(distance) or distance <= 0):
            return uncertain(f"Missing or invalid calibrated measurement: {name}")
        values[name] = distance
    candidates = []
    for size, reference in REFERENCES.items():
        deltas = {name: values[name] - expected for name, expected in reference.items()}
        candidates.append({
            "size": size,
            "rmse_in": math.sqrt(sum(d * d for d in deltas.values()) / len(deltas)),
            "within_tolerance": all(abs(d) <= tolerance_in + 1e-9 for d in deltas.values()),
            "deltas_in": deltas,
        })
    candidates.sort(key=lambda candidate: candidate["rmse_in"])
    matches = [candidate for candidate in candidates if candidate["within_tolerance"]]
    if len(matches) == 1:
        result = {"status": "matched", "size": matches[0]["size"],
                  "reason": "All four measurements match one supplied reference."}
    elif matches:
        result = uncertain("Measurements overlap multiple references; inspect manually.")
    else:
        result = uncertain("No reference matches all four measurements; inspect manually.")
    closest = candidates[0]
    return {**result, "tolerance_in": tolerance_in, "candidates": candidates,
            "reference": REFERENCES[result["size"]] if result["size"] else None,
            # Always populated (even when uncertain) so a manual reviewer sees how close the
            # nearest size actually was, instead of a blank comparison column.
            "closest_size": closest["size"], "closest_reference": REFERENCES[closest["size"]],
            "closest_deltas_in": closest["deltas_in"], "closest_rmse_in": closest["rmse_in"]}


def check_inspection_mask(mask):
    """Reject obvious empty, crowded, or edge-clipped segmentation masks.

    These heuristic checks are not proof that the scene contains underwear.
    Caller must restrict the inspection station to a single, flat garment.
    """
    if mask is None or np.asarray(mask).ndim != 2:
        raise InspectionRejected("No usable garment mask.")
    mask = np.asarray(mask)
    binary = (mask > (0.5 if mask.max() <= 1 else 127)).astype(np.uint8)
    height, width = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = sorted(stats[1:], key=lambda row: row[cv2.CC_STAT_AREA], reverse=True)
    if count < 2 or components[0][cv2.CC_STAT_AREA] < binary.size * 0.03:
        raise InspectionRejected("No clear garment; place one item in the inspection area.")
    x, y, w, h, area = (int(v) for v in components[0])
    if area > binary.size * 0.90:
        raise InspectionRejected("Garment fills the view; leave visible background around it.")
    if len(components) > 1 and components[1][cv2.CC_STAT_AREA] > max(binary.size * .015, area * .08):
        raise InspectionRejected("More than one object detected; remove hands, ruler and other items.")
    margin = max(2, round(min(height, width) * .01))
    if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
        raise InspectionRejected("Garment touches the inspection boundary; show the entire item.")


def check_inspection_landmarks(landmarks):
    """Basic geometry gate for the upright shorts landmark model used by this fork."""
    try:
        points = {key: (float(landmarks[key]["x"]), float(landmarks[key]["y"]))
                  for key in ("1", "2", "3", "4", "5", "7", "10")}
    except (KeyError, TypeError, ValueError):
        raise InspectionRejected("Incomplete garment landmarks.") from None
    if not all(math.isfinite(v) for point in points.values() for v in point):
        raise InspectionRejected("Invalid garment landmarks.")
    left, center, right = (points[key] for key in ("1", "2", "3"))
    waist_width = right[0] - left[0]
    if (waist_width <= 0 or abs(right[1] - left[1]) > waist_width * .20
            or not left[0] < center[0] < right[0]
            or points["5"][1] <= left[1] or points["7"][1] <= center[1]
            or points["4"][0] >= points["10"][0]):
        raise InspectionRejected("Lay the item flat with its waistband horizontal at the top.")
