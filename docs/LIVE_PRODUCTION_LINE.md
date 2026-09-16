# Top-down underwear inspection (items pause)

This mode estimates **M, L, XL or 2XL** from a fixed overhead camera using your
four underwear reference screenshots. Each item pauses, the page waits for a
stable view, and the server measures a single snapshot. There is no size selector:
the measured dimensions determine the estimate. No additional dependencies or
model downloads are needed if the existing app already runs.

This is an inspection prototype, not a validated production sorter. It does not
control the conveyor, trigger a PLC, count items reliably, or train a new neural
network. The existing **shorts landmark model** measures underwear; reference
matching then makes the size decision. Use it only for the same garment style,
laid out consistently. Other styles and sizes need their own verified references.

## Start on the PC

Stop the photo-upload server first (both shortcuts default to port 8080):

```powershell
.\run_phone_pc.bat
```

Wait for the server's `Running on http://127.0.0.1:8080` message, then open that
address in Chrome or Edge. Allow camera access and choose the overhead camera
from the page's Camera menu. USB cameras/capture devices exposed to the browser
are supported; the page does not open an RTSP/IP-camera URL directly.

For the other PC, complete [GTX 1050 Ti setup](GTX1050TI_SETUP.md) first. The same
launcher uses the GPU automatically when working CUDA is available. The new
inspection mode does not change those GPU settings. To require CUDA:

```powershell
.\run_phone_pc.bat --device cuda
```

For a phone camera, use the HTTPS LAN launcher and instructions in that guide.
A phone must be fixed overhead too; handheld movement invalidates calibration.

## Set up the station

1. Fix the camera directly above the inspection surface, parallel to it. Use
   even lighting, a plain contrasting belt, and no camera motion, digital zoom,
   autofocus breathing, or changing exposure if your camera allows these to be
   locked. Keep the whole garment in view with visible background on every side.
2. Put the steel ruler at garment height. Click **1. Calibrate 12-inch ruler**.
   On the frozen view click the **0-inch mark**, then the **12-inch mark**, not
   the outer metal edges. The selected span must be at least 60 pixels.
3. Optionally click **2. Set inspection area** and click opposite corners of a
   rectangle around the station. Include the largest garment plus margin; exclude
   belt edges, neighbouring items and operators' hands. The default is the whole
   camera frame. Cropping does not rescale pixels, so it preserves calibration.
4. Remove the ruler, garment and hands. With the inspection area completely
   empty, click **3. Capture empty inspection area**. Do not capture this reference
   while a garment is present. Changing the area or calibration clears it.
5. Click **Start inspection**. Put one underwear item flat and unstretched in
   the area, front side up, waistband horizontal at the **top** of the image.
6. Let the item pause. After about **1.25 seconds of a stable view**, one frame
   is submitted. **Hold it still until the result appears.** Total dwell time
   must include model inference and transfer time, shown beside the snapshot.
   CPU inference can be much slower than the line; GTX 1050 Ti throughput still
   needs measurement on your actual PC. No video-rate inference is promised.
7. Remove the item. Leave the inspection area visibly empty for **at least one
   second** before the next one. This gap rearms the trigger. If you need to retry
   the same item, click **Reinspect item** while it is paused.

The live video stays live. The annotated image is a separate, timestamped
**inspected snapshot**, never an overlay pretending to track the moving garment.
Movement/removal clears the current result and invalidates any pending response
for that item. Only one request is outstanding per page; a busy server rejects
other requests rather than queueing old frames. Stop prevents new requests and
discards pending results; it does not cancel inference already running on the PC.
Requests time out after five minutes, without interrupting the server's model.

Keep the page visible. Hiding the tab stops inspection. Camera disconnection,
resolution changes or a frozen video reset setup. Physical camera movement,
zoom, belt-height changes and lighting changes cannot all be detected reliably:
stop, recalibrate and capture a new empty reference after any such change.
Single-frame measurement for other clothing remains in the Inspection mode menu.

## References and decision rule

These are the **measured** values from your screenshots, not the unrelated old
"Expected" shorts ranges. They are laid-flat garment dimensions in inches, not
body measurements or waist/hip circumferences. XXL is labelled `2XL`.

| Size | Front rise | Full length | Hips (flat) | Waist (flat) |
| --- | ---: | ---: | ---: | ---: |
| M | 11.24 | 11.36 | 14.62 | 13.69 |
| L | 11.14 | 12.65 | 16.72 | 14.62 |
| XL | 11.30 | 12.16 | 17.05 | 15.79 |
| 2XL | 12.28 | 12.39 | 18.09 | 16.84 |

The code preserves these values exactly, including non-monotonic lengths.
It accepts a size only if **all four measurements** are within **±0.50 inches**
of exactly one reference. No match, multiple matches or invalid measurements
produce **Uncertain**, not a forced nearest size. The allowance is provisional,
not a manufacturing tolerance or statistical confidence interval. Candidate
distances in the API are RMSE in inches, not confidence percentages.

References and the allowance live in
`src/garmentiq/underwear_sizes.py` (`REFERENCES`, `DEFAULT_TOLERANCE_IN`). Restart
the server after changing them. The live reference table reads the server's data.
The old photo-upload page retains its existing manual size-comparison UI.

## Quality checks and limitations

- Browser probes run four times per second on a small RGB preview. Presence
  means at least 5% of preview pixels differ from the empty reference; stillness
  requires no more than 1.5% changing pixels (average RGB difference over 22/255).
  An anchor comparison also checks cumulative drift. Empty rearming needs 750 ms.
  These are heuristics in `phone_web/inspection.mjs`, not a trained object tracker.
- The server rejects obviously empty, crowded, frame-filling or edge-clipped
  masks, and invalid/basic misaligned landmark geometry. Small mask noise is
  ignored. These checks cannot prove an object is underwear or reliably reject
  overlapping garments, folds, blur, hands or a convincing false segmentation.
- Patterned/moving belts, shadows and lighting flicker can confuse presence and
  stillness. A nearly identical replacement without a visible empty gap can be
  missed. This mode requires separated, paused items and a clear station.
- A one-dimensional ruler scale assumes a flat scene and negligible perspective
  and lens distortion. It is not a full camera calibration. Verify scale at the
  centre and edges with physical measurements before using the whole area.
- One sample per size is not enough to establish brand/style variation or model
  accuracy. Validate on multiple independently labelled garments per size,
  repeating measurements across placement and lighting conditions. Check actual
  dimensions with a ruler/tape and record wrong-size and uncertain outcomes.
  Do not widen tolerances just to turn failures into matches.

Before connecting this to automatic sorting, test the actual camera, fabric,
lighting and GPU; collect representative labelled line images if the landmark
model needs adaptation. Choose factory tolerances using those results and define
a manual-review path for uncertain/failed items. PLC triggers, reject gates,
traceable logs and production-server hardening are separate integration work.
The bundled Flask server is a trusted-local-network demo, not an internet service.

## Developer checks

From the repository root, using the installed environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s test -p 'test_*.py' -v
node --test test/test_inspection.mjs
node --test test/test_live_page.mjs
```

Python tests cover reference decisions, invalid calibration, frame-quality
rejection, real measurement math with a stub detector, API compatibility and
busy/error handling. Node tests cover the pause/empty/next-item trigger and stale
result invalidation, plus page behavior with simulated camera/DOM adapters. These
do not replace visual browser testing, a real camera or a production accuracy
trial. Node is only needed for development tests, not to run the app.
