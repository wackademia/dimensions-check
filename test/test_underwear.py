"""Reference matching, frame safety and live HTTP integration (no model downloads)."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import phone_server
from garmentiq.live import LiveDetector
from garmentiq.underwear_sizes import (
    REFERENCES, InspectionRejected, check_inspection_landmarks,
    check_inspection_mask, estimate_size,
)


def measurements(size):
    return {name: {"distance": value, "unit": "in"} for name, value in REFERENCES[size].items()}


class SizeTests(unittest.TestCase):
    def test_exact_reference_sizes(self):
        for size in REFERENCES:
            with self.subTest(size=size):
                result = estimate_size(measurements(size))
                self.assertEqual(result["size"], size)
                self.assertEqual(result["status"], "matched")
                self.assertEqual(result["candidates"][0]["rmse_in"], 0)

    def test_known_reference_values_are_preserved(self):
        self.assertEqual(REFERENCES["M"], {"front rise": 11.24, "full length": 11.36, "hips": 14.62, "waist": 13.69})
        self.assertEqual(REFERENCES["L"]["full length"], 12.65)
        self.assertEqual(REFERENCES["XL"]["waist"], 15.79)
        self.assertEqual(REFERENCES["2XL"]["hips"], 18.09)

    def test_tolerance_boundary_and_no_forced_nearest_size(self):
        sample = measurements("M")
        sample["waist"]["distance"] -= .50
        self.assertEqual(estimate_size(sample)["size"], "M")
        sample["waist"]["distance"] -= .01
        self.assertIsNone(estimate_size(sample)["size"])

    def test_non_waist_dimensions_do_not_affect_size(self):
        sample = measurements("XL")
        sample["full length"]["distance"] = 25
        self.assertEqual(estimate_size(sample)["size"], "XL")

    def test_ambiguous_is_not_forced(self):
        sample = {key: {"distance": (REFERENCES["L"][key] + REFERENCES["XL"][key]) / 2, "unit": "in"} for key in REFERENCES["L"]}
        self.assertEqual(estimate_size(sample, .7)["status"], "uncertain")

    def test_uncertain_result_still_reports_the_closest_size(self):
        sample = measurements("XL")
        sample["waist"]["distance"] = 25
        result = estimate_size(sample)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["reference"])
        self.assertEqual(result["closest_size"], result["candidates"][0]["size"])
        self.assertEqual(result["closest_reference"], REFERENCES[result["closest_size"]])
        self.assertEqual(result["closest_deltas_in"], result["candidates"][0]["deltas_in"])
        self.assertEqual(result["closest_rmse_in"], result["candidates"][0]["rmse_in"])

    def test_matched_result_closest_fields_agree_with_the_match(self):
        result = estimate_size(measurements("M"))
        self.assertEqual(result["closest_size"], "M")
        self.assertEqual(result["closest_reference"], result["reference"])

    def test_invalid_or_missing_measurements(self):
        for value in (float("nan"), float("inf"), 0, -1, "14", True, None):
            sample = measurements("M")
            sample["waist"]["distance"] = value
            self.assertIsNone(estimate_size(sample)["size"])
        sample = measurements("M")
        sample["waist"]["unit"] = "px"
        self.assertIsNone(estimate_size(sample)["size"])
        self.assertIsNone(estimate_size({})["size"])

    def test_bad_tolerance(self):
        for tolerance in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                estimate_size(measurements("M"), tolerance)


def fake_agent():
    schema = json.loads((Path(__file__).parents[1] / "src/garmentiq/instruction/shorts.json").read_text())
    coordinates = {"1": (50, 30), "2": (123.1, 30), "3": (196.2, 30),
                   "4": (40, 90), "5": (50, 156.5), "7": (123.1, 141.4), "10": (207.2, 90)}
    for key, (x, y) in coordinates.items():
        schema["shorts"]["landmarks"][key].update(x=x, y=y, conf=.9)
    mask = np.zeros((240, 300), np.uint8)
    mask[20:190, 30:220] = 255
    return SimpleNamespace(
        do_refine=False, do_derive=False, segmentation_model_args={"background_color": [102, 255, 102]},
        segment=lambda rgb: (rgb, mask.copy(), rgb),
        detect=lambda label, rgb: (None, None, copy.deepcopy(schema)),
    )


class FrameTests(unittest.TestCase):
    def test_empty_full_clipped_crowded_rejected(self):
        empty = np.zeros((200, 200), np.uint8)
        full = np.full((200, 200), 255, np.uint8)
        clipped = empty.copy(); clipped[0:120, 20:140] = 255
        crowded = empty.copy(); crowded[20:180, 20:70] = 255; crowded[20:180, 120:170] = 255
        for mask in (None, empty, full, clipped, crowded):
            with self.assertRaises(InspectionRejected):
                check_inspection_mask(mask)

    def test_centered_mask_and_small_noise_allowed(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:180, 20:180] = 255; mask[1, 1] = 255
        check_inspection_mask(mask)
        check_inspection_mask(mask / 255)

    def test_invalid_landmarks(self):
        with self.assertRaises(InspectionRejected):
            check_inspection_landmarks({})
        agent = fake_agent()
        landmarks = agent.detect("shorts", None)[2]["shorts"]["landmarks"]
        landmarks["3"]["y"] += 100
        with self.assertRaises(InspectionRejected):
            check_inspection_landmarks(landmarks)

    def test_inspection_requires_short_model_and_inches(self):
        with self.assertRaises(ValueError):
            LiveDetector(fake_agent(), inspect_underwear=True)
        for scale in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                LiveDetector(fake_agent(), pixels_per_unit=scale)

    def test_real_measurement_math_to_size(self):
        result = LiveDetector(fake_agent(), class_name="shorts", pixels_per_unit=10,
                              unit_name="in", inspect_underwear=True).process_frame(np.zeros((240, 300, 3), np.uint8))
        self.assertEqual(estimate_size(result.measurements)["size"], "L")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = phone_server.app.test_client()
        self.agent_patch = patch.object(phone_server, "agent", fake_agent())
        self.agent_patch.start()
        self.addCleanup(self.agent_patch.stop)
        _, image = cv2.imencode(".jpg", np.zeros((240, 300, 3), np.uint8))
        self.image = image.tobytes()

    def post(self, **fields):
        return self.client.post('/measure', data={"frame": (io.BytesIO(self.image), "frame.jpg"),
            "pixels_per_inch": "10", "mode": "underwear", **fields})

    def test_auto_underwear_forces_shorts_and_estimates_size(self):
        response = self.post(class_name="vest dress")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["garment_class"], "shorts")
        self.assertEqual(data["size_estimate"]["size"], "L")
        self.assertTrue(data["image"].startswith("data:image/jpeg;base64,"))

    def test_legacy_mode_still_measures_without_size_estimate(self):
        response = self.post(mode="", class_name="shorts")
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.get_json()["size_estimate"])

    def test_uncertain_mask_response_not_server_error(self):
        with patch.object(phone_server.agent, "segment", return_value=(None, np.zeros((240, 300), np.uint8), None)):
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["size_estimate"]["status"], "uncertain")
        self.assertEqual(response.get_json()["measurements"], {})

    def test_invalid_scale_class_mode_and_frames(self):
        for scale in ("nan", "inf", "0", "-1", "abc", ""):
            self.assertEqual(self.post(pixels_per_inch=scale).status_code, 400)
        self.assertEqual(self.post(class_name="invalid").status_code, 400)
        self.assertEqual(self.post(mode="invalid").status_code, 400)
        self.assertEqual(self.client.post('/measure').status_code, 400)
        self.image = b""
        self.assertEqual(self.post().status_code, 400)
        self.image = b"invalid"
        self.assertEqual(self.post().status_code, 400)

    def test_busy_returns_429_without_queueing(self):
        with phone_server.inference_lock:
            self.assertEqual(self.post().status_code, 429)
        self.assertEqual(self.post().status_code, 200)

    def test_failure_releases_lock(self):
        with patch.object(phone_server.agent, "detect", side_effect=RuntimeError("test failure")):
            with self.assertLogs(phone_server.app.logger, level="ERROR"):
                self.assertEqual(self.post().status_code, 500)
        self.assertFalse(phone_server.inference_lock.locked())
        self.assertEqual(self.post().status_code, 200)

    def test_references_and_frontend_assets(self):
        self.assertEqual(self.client.get('/size-references').get_json()["references"], REFERENCES)
        for url in ('/', '/phone_web/app.js', '/phone_web/inspection.mjs', '/phone_web/style.css'):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                if url.endswith('.mjs'):
                    self.assertIn(response.mimetype, ('text/javascript', 'application/javascript'))
                response.close()


if __name__ == '__main__':
    unittest.main()
