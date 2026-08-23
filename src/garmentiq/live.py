"""Live video garment measurement using an already configured ``tailor`` agent."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import cv2
import numpy as np

from . import utils


@dataclass
class LiveResult:
    frame: np.ndarray
    garment_class: str
    measurements: dict
    detection: dict
    inference_seconds: float


class LiveDetector:
    """Run the image pipeline repeatedly on RGB video frames.

    ``pixels_per_unit`` converts the model's pixel distances to a real unit. For
    example, if a 10 cm reference spans 250 pixels, pass ``25`` and
    ``unit_name="cm"``. Without calibration, measurements remain in pixels.
    """

    def __init__(
        self,
        agent,
        *,
        pixels_per_unit: Optional[float] = None,
        unit_name: str = "px",
        class_name: Optional[str] = None,
    ):
        if pixels_per_unit is not None and pixels_per_unit <= 0:
            raise ValueError("pixels_per_unit must be greater than zero")
        self.agent = agent
        self.pixels_per_unit = pixels_per_unit
        self.unit_name = unit_name if pixels_per_unit else "px"
        self.class_name = class_name

    def process_frame(self, frame_bgr: np.ndarray) -> LiveResult:
        """Measure one OpenCV BGR frame and return an annotated BGR frame."""
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("frame_bgr must be a HxWx3 NumPy array")

        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        label = self.class_name or self.agent.classify(rgb, verbose=False)[0]
        needs_mask = self.agent.do_derive or self.agent.do_refine
        use_bg = self.agent.segmentation_model_args.get("background_color") is not None

        mask = None
        detection_image = rgb
        if needs_mask or use_bg:
            segmented = self.agent.segment(rgb)
            mask = segmented[1]
            if use_bg:
                detection_image = segmented[2]

        coords, confidences, detection = self.agent.detect(label, detection_image)
        if self.agent.do_refine:
            coords, detection = self.agent.refine(
                label, coords, confidences, detection, mask
            )
        if self.agent.do_derive:
            _, detection = self.agent.derive(
                label, detection, self.agent.derivation_dict, coords, mask
            )

        cleaned = utils.clean_detection_dict(label, "live_frame", detection)
        _, cleaned = utils.compute_measurement_distances(cleaned)
        garment = cleaned["live_frame"]
        measurements = copy.deepcopy(garment["measurements"])
        if self.pixels_per_unit:
            for measurement in measurements.values():
                measurement["distance"] /= self.pixels_per_unit
                measurement["unit"] = self.unit_name
        else:
            for measurement in measurements.values():
                measurement["unit"] = "px"

        annotated = self._draw(frame_bgr.copy(), garment["landmarks"], measurements)
        return LiveResult(
            frame=annotated,
            garment_class=label,
            measurements=measurements,
            detection=detection,
            inference_seconds=time.perf_counter() - started,
        )

    def _draw(self, frame, landmarks, measurements):
        for landmark in landmarks.values():
            cv2.circle(frame, (round(landmark["x"]), round(landmark["y"])), 5, (0, 255, 0), -1)
        y = 28
        for name, value in measurements.items():
            start = landmarks[value["landmarks"]["start"]]
            end = landmarks[value["landmarks"]["end"]]
            p1 = (round(start["x"]), round(start["y"]))
            p2 = (round(end["x"]), round(end["y"]))
            cv2.line(frame, p1, p2, (0, 220, 255), 2)
            text = f'{name}: {value["distance"]:.2f} {value["unit"]}'
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            y += 26
        return frame

    def run(
        self,
        source: Union[int, str] = 0,
        *,
        process_every_n_frames: int = 1,
        window_name: str = "GarmentIQ Live",
        on_result: Optional[Callable[[LiveResult], None]] = None,
    ) -> None:
        """Open a camera/video and process frames until Q or Escape is pressed."""
        if process_every_n_frames < 1:
            raise ValueError("process_every_n_frames must be at least 1")
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video source: {source!r}")

        last_frame = None
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % process_every_n_frames == 0:
                    result = self.process_frame(frame)
                    last_frame = result.frame
                    if on_result:
                        on_result(result)
                cv2.imshow(window_name, last_frame if last_frame is not None else frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                frame_index += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()
