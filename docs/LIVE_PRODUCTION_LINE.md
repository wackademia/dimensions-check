# Top-down underwear inspection (present, then press Start)

This mode estimates **M, L, XL or 2XL** from a fixed overhead camera using your
four underwear reference screenshots. The operator places one item, presses
**Start inspection**, the page waits briefly for a stable view, and the server
measures a single snapshot. There is no size selector: the measured dimensions
determine the estimate. No additional dependencies or model downloads are
needed if the existing app already runs.

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

On startup the server prompts in the terminal for a folder to save inspection
snapshots to (press Enter to accept the default `snapshots` folder next to the
server). Every underwear inspection that produces measurements and a size
decision saves one JPEG there: the annotated photo followed by a table of each
measurement's value against a reference and its delta. When a size matches,
that's the matched reference; when the result is **Uncertain**, the table still
shows the *closest* size and how far off each dimension was, labelled
"unmatched", instead of leaving the comparison blank. This runs regardless of
the `--device`/GPU flags above.

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
4. Place one underwear item flat and unstretched in the area, front side up,
   waistband horizontal at the **top** of the image, then click **Start
   inspection**.
5. The page waits for about **1.25 seconds of a stable view** (so the captured
   frame isn't mid-motion), then submits one snapshot automatically — no
   further click needed. **Start inspection** stays disabled while this runs.
   Total time includes model inference and transfer time, shown beside the
   snapshot. CPU inference can be much slower than the line; GTX 1050 Ti
   throughput still needs measurement on your actual PC. No video-rate
   inference is promised.
6. The result (size, measurements and the closest reference) stays on screen
   indefinitely — nothing clears it automatically. Remove the inspected item,
   place the next one, and click **Start inspection** again whenever you're
   ready; that click is what replaces the displayed result with the new item's.
   If a capture fails or you want to redo the current item, just click **Start
   inspection** again.

The live video stays live. The annotated image is a separate, timestamped
**inspected snapshot**, never an overlay pretending to track the moving garment.
Only one request is outstanding per page; a busy server rejects other requests
rather than queueing old frames. Because the page waits for you to press the
button, there is no automatic presence/removal detection to confuse — the only
thing that can go wrong before capture is the stillness check, which just waits
longer if the item (or the camera) is still moving.

Keep the page visible. Hiding the tab while a capture is in progress aborts the
wait and requires pressing **Start inspection** again; it does not cancel
inference already running on the PC if the request had already been sent.
Camera disconnection, resolution changes or a frozen video reset setup.
Physical camera movement, zoom, belt-height changes and lighting changes cannot
all be detected reliably: recalibrate after any such change. Single-frame
measurement for other clothing remains in the Inspection mode menu.

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
It accepts a size only if the **flat waist measurement** is within **±0.50 inches**
of exactly one reference. No match, multiple matches or an invalid waist measurement
produce **Uncertain**, not a forced nearest size. The allowance is provisional,
not a manufacturing tolerance or statistical confidence interval. Candidate
distances in the API use the legacy `rmse_in` field name for compatibility, but
the value is now the absolute waist delta in inches, not a confidence percentage.

References and the allowance live in
`src/garmentiq/underwear_sizes.py` (`REFERENCES`, `DEFAULT_TOLERANCE_IN`). Restart
the server after changing them. The live reference table reads the server's data.
The old photo-upload page retains its existing manual size-comparison UI.

## Quality checks and limitations

- After you click **Start inspection**, the browser polls a small RGB preview
  four times per second purely to detect *stillness* — no more than 1.5%
  changing pixels (average RGB difference over 22/255) for about 1.25 seconds
  — so the captured frame isn't blurry. An anchor comparison also checks
  cumulative drift. This is a heuristic in `phone_web/inspection.mjs`
  (`SettleGate`), not a trained object tracker, and it does **not** try to
  detect whether an item is present — that's the operator's job (only press
  the button when something is actually in view) and the server's job (see
  next bullet).
- The server rejects obviously empty, crowded, frame-filling or edge-clipped
  masks, and invalid/basic misaligned landmark geometry. Small mask noise is
  ignored. These checks cannot prove an object is underwear or reliably reject
  overlapping garments, folds, blur, hands or a convincing false segmentation.
  If the button is pressed with nothing in the area, expect an **Uncertain**
  result with a reason like "No clear garment; place one item in the
  inspection area." rather than a crash.
- Patterned/moving belts, shadows, lighting flicker, or the operator's hand
  lingering in frame can all register as "still" and get captured — the
  stillness check only measures RGB change, not what's actually in view.
- A one-dimensional ruler scale assumes a flat scene and negligible perspective
  and lens distortion. It is not a full camera calibration. Verify scale at the
  centre and edges with physical measurements before using the whole area.
- One sample per size is not enough to establish brand/style variation or model
  accuracy. Validate on multiple independently labelled garments per size,
  repeating measurements across placement and lighting conditions. Check actual
  dimensions with a ruler/tape and record wrong-size and uncertain outcomes.
  Do not widen tolerances just to turn failures into matches. **Uncertain is an
  expected, common outcome of this design** — it means no single reference
  matched the waist width within tolerance, not that something is broken.
  The result panel and saved snapshot both show the *closest* size and its
  per-dimension deltas even when uncertain, specifically so a manual reviewer
  has real numbers to judge from instead of a blank "no match."

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

Python tests cover reference decisions (including the closest-match fields
reported when uncertain), invalid calibration, frame-quality rejection, real
measurement math with a stub detector, API compatibility and busy/error
handling. Node tests cover the settle-then-capture stillness trigger and stale
result invalidation, plus page behavior with simulated camera/DOM adapters. These
do not replace visual browser testing, a real camera or a production accuracy
trial. Node is only needed for development tests, not to run the app.
