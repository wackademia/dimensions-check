import {SettleGate} from './inspection.mjs';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), context = overlay.getContext('2d');
const probe = document.createElement('canvas');
probe.width = 96; probe.height = 72;
const probeContext = probe.getContext('2d', {willReadFrequently: true});
let stream, roi, settleTimer, frozen;
let scale = 0, selection = null, points = [], busy = false, epoch = 0;
let cameraReady = false, lastVideoTime = -1, lastVideoAdvance = 0;
const isUnderwear = () => $('mode').value === 'underwear';
const say = text => { $('status').textContent = text; };

function clearResult(title = 'Waiting') {
  $('size').textContent = title;
  $('size').className = 'size';
  $('resultPanel').hidden = true;
  $('result').removeAttribute('src');
  $('measurements').replaceChildren();
}
function controls() {
  for (const id of ['camera', 'mode', 'className', 'calibrate', 'area']) {
    $(id).disabled = busy || !!selection || !cameraReady;
  }
  $('start').disabled = !cameraReady || !scale || busy || !!selection;
  $('measure').disabled = !cameraReady || !scale || busy || !!selection;
  $('measure').hidden = isUnderwear();
  $('classControl').hidden = isUnderwear();
  $('start').hidden = !isUnderwear();
  $('cancel').hidden = !selection;
}
function drawArea() {
  context.clearRect(0, 0, overlay.width, overlay.height);
  if (!roi) return;
  context.strokeStyle = '#81e6b5'; context.lineWidth = 3;
  context.strokeRect(roi.x + 1, roi.y + 1, roi.w - 2, roi.h - 2);
}
// Bumps epoch (so any in-flight capture/settle cycle discards its result rather than
// showing something stale) and, if we're still in the pre-capture settle wait (nothing
// sent to the server yet), frees the Start button immediately. A request already sent to
// the server keeps `busy` set until its own response/failure resolves — this does not
// cancel inference already running on the PC, only what happens with its result.
function abortInspection(message) {
  epoch++;
  if (settleTimer) { clearInterval(settleTimer); settleTimer = null; busy = false; }
  clearResult('Stopped');
  if (message) say(message);
  controls();
}
function resetSetup(message) {
  abortInspection(message); scale = 0; selection = null;
  $('setup').textContent = message;
  roi = {x: 0, y: 0, w: video.videoWidth, h: video.videoHeight};
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  drawArea(); controls();
}
async function startCamera(deviceId = '') {
  cameraReady = false;
  abortInspection('Connecting to camera...');
  stream?.getTracks().forEach(track => track.stop());
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Use localhost on the PC or HTTPS on a phone.');
    stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {
      ...(deviceId ? {deviceId: {exact: deviceId}} : {facingMode: 'environment'}),
      width: {ideal: 1280}, height: {ideal: 960},
    }});
    video.srcObject = stream;
    await video.play();
    cameraReady = true;
    resetSetup('Camera ready. Calibrate using the 0 and 12-inch ruler marks.');
    $('cameraStatus').textContent = `Live camera · ${video.videoWidth} × ${video.videoHeight}. The green box is the inspection area.`;
    const cameras = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    $('camera').replaceChildren(...cameras.map((d, i) => new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
    $('camera').value = stream.getVideoTracks()[0].getSettings().deviceId || '';
    stream.getVideoTracks()[0].onended = () => {
      cameraReady = false; resetSetup('Camera disconnected. Reload this page to reconnect and recalibrate.');
    };
  } catch (error) {
    cameraReady = false; scale = 0;
    say(`Camera error: ${error.message}`); $('cameraStatus').textContent = 'No camera available. Check its connection and browser permission.';
    controls(); $('camera').disabled = false;
  }
}
function freezeFor(kind) {
  epoch++; selection = kind; points = []; clearResult('Setup');
  if (kind === 'ruler') scale = 0;
  frozen = document.createElement('canvas'); frozen.width = video.videoWidth; frozen.height = video.videoHeight;
  frozen.getContext('2d').drawImage(video, 0, 0);
  context.drawImage(frozen, 0, 0);
  say(kind === 'ruler' ? 'Frozen view: click the 0-inch mark, then the 12-inch mark.' : 'Frozen view: click two opposite corners of the inspection area. Leave margin around the whole garment.');
  controls();
}
function snapshot() {
  const shot = document.createElement('canvas'); shot.width = roi.w; shot.height = roi.h;
  shot.getContext('2d').drawImage(video, roi.x, roi.y, roi.w, roi.h, 0, 0, roi.w, roi.h);
  return shot;
}
function sample() {
  probeContext.drawImage(video, roi.x, roi.y, roi.w, roi.h, 0, 0, probe.width, probe.height);
  return probeContext.getImageData(0, 0, probe.width, probe.height).data;
}
overlay.addEventListener('pointerdown', event => {
  if (!selection) return;
  const rect = overlay.getBoundingClientRect();
  const point = {x: Math.min(overlay.width, Math.max(0, Math.round((event.clientX - rect.left) * overlay.width / rect.width))),
                 y: Math.min(overlay.height, Math.max(0, Math.round((event.clientY - rect.top) * overlay.height / rect.height)))};
  points.push(point); context.fillStyle = '#81e6b5'; context.beginPath(); context.arc(point.x, point.y, 7, 0, 2 * Math.PI); context.fill();
  if (points.length < 2) return;
  const [a, b] = points;
  if (selection === 'ruler') {
    const distance = Math.hypot(b.x - a.x, b.y - a.y);
    if (distance < 60) { points = []; context.drawImage(frozen, 0, 0); say('Ruler marks are too close. Select marks at least 60 pixels apart.'); return; }
    scale = distance / 12;
  } else {
    const w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
    if (Math.min(w, h) < 64) { points = []; context.drawImage(frozen, 0, 0); say('Inspection area is too small. Select a larger area.'); return; }
    roi = {x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w, h};
  }
  selection = null; frozen = null; drawArea(); controls();
  $('setup').textContent = scale ? `Calibrated: ${scale.toFixed(2)} pixels/inch. Area: ${roi.w} × ${roi.h} pixels.` : 'Calibrate the ruler before continuing.';
  say('Remove the ruler, then present an item and press Start inspection.');
});

function renderResult(data, capturedAt, elapsed) {
  const decision = data.size_estimate;
  $('size').textContent = decision ? (decision.size ? `Estimated ${decision.size}` : 'Uncertain') : data.garment_class;
  $('size').className = `size ${decision?.status || ''}`;
  say(decision ? decision.reason : 'Measurement complete. Values below are in inches.');
  $('timing').textContent = `Inspected snapshot: ${capturedAt.toLocaleTimeString()} · model ${data.inference_seconds}s · round trip ${elapsed.toFixed(1)}s. This is not the current video frame.`;
  $('result').src = data.image; $('resultPanel').hidden = false;
  // A confirmed match shows its reference; an uncertain result still shows the closest
  // size's reference and how far off each measurement was, instead of a blank column.
  const matched = decision?.reference;
  const closest = decision?.closest_reference;
  $('referenceHeader').textContent = matched ? 'Reference'
    : closest ? `Closest: ${decision.closest_size} (unmatched)` : 'Reference';
  $('measurements').replaceChildren(...Object.entries(data.measurements).map(([name, value]) => {
    const row = document.createElement('tr');
    const reference = matched?.[name] ?? closest?.[name];
    const referenceText = reference == null ? '—'
      : `${reference.toFixed(2)} in (Δ${(value.distance - reference >= 0 ? '+' : '')}${(value.distance - reference).toFixed(2)})`;
    for (const text of [name, `${value.distance.toFixed(2)} in`, referenceText]) {
      const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
    }
    return row;
  }));
}
async function captureAndMeasure() {
  const requestEpoch = epoch;
  const current = () => requestEpoch === epoch;
  clearResult('Inspecting'); say('Measuring the current frame...');
  const capturedAt = new Date(), started = performance.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300000);
  try {
    const shot = snapshot(); // Crop only: no rescaling, so ruler pixels/inch stays valid.
    const blob = await new Promise(resolve => shot.toBlob(resolve, 'image/jpeg', .92));
    if (!blob) throw new Error('Could not capture camera frame.');
    if (!current()) return;
    const form = new FormData(); form.append('frame', blob, 'inspection.jpg');
    form.append('pixels_per_inch', scale); form.append('mode', $('mode').value);
    form.append('class_name', isUnderwear() ? 'shorts' : $('className').value);
    const response = await fetch('/measure', {method: 'POST', body: form, signal: controller.signal});
    const data = await response.json();
    if (!current()) return; // Never put a departed item's result on a newer item.
    if (!response.ok) throw new Error(data.error || `Server returned ${response.status}`);
    renderResult(data, capturedAt, (performance.now() - started) / 1000);
  } catch (error) {
    if (current()) {
      clearResult('Inspection failed');
      say(`${error.name === 'AbortError' ? 'Request timed out; the server may still be processing.' : error.message} Press Start inspection to retry.`);
    }
  } finally {
    clearTimeout(timeout);
  }
}
// One press = one item: wait briefly for it to stop moving (so the captured frame isn't
// blurry), then measure it once. The result stays on screen — nothing here clears it —
// until the next press overwrites it with a new one.
function startInspection() {
  if (busy || selection) return;
  busy = true; controls();
  const requestEpoch = epoch;
  const gate = new SettleGate();
  let lastState = '';
  lastVideoTime = -1; lastVideoAdvance = performance.now();
  say('Hold the item still...');
  settleTimer = setInterval(() => {
    if (requestEpoch !== epoch) return; // aborted elsewhere; that already cleared this timer
    if (!cameraReady || video.readyState < 2 || video.videoWidth !== overlay.width || video.videoHeight !== overlay.height) {
      resetSetup('Video changed or became unavailable. Recalibrate and try again.'); return;
    }
    const now = performance.now();
    if (video.currentTime !== lastVideoTime) { lastVideoTime = video.currentTime; lastVideoAdvance = now; }
    else if (now - lastVideoAdvance > 2000) { resetSetup('Camera feed froze. Check the camera, then try again.'); return; }
    const state = gate.update(sample(), now);
    $('activity').textContent = `motion ${(state.motion * 100).toFixed(1)}%`;
    if (state.state !== lastState) {
      const messages = {moving: 'Item is moving. Waiting for it to settle.', settling: 'Waiting for a stable view.'};
      if (messages[state.state]) say(messages[state.state]);
      lastState = state.state;
    }
    if (state.state === 'ready') {
      clearInterval(settleTimer); settleTimer = null;
      captureAndMeasure().finally(() => { busy = false; controls(); });
    }
  }, 250);
}
$('camera').onchange = () => void startCamera($('camera').value);
$('calibrate').onclick = () => freezeFor('ruler');
$('area').onclick = () => freezeFor('area');
$('cancel').onclick = () => { selection = null; frozen = null; drawArea(); controls(); say('Selection cancelled.'); };
$('mode').onchange = () => {
  abortInspection(); clearResult('Ready');
  say(isUnderwear() ? 'Present an item, then press Start inspection.' : 'Select a garment type and measure a single frame.');
  controls();
};
$('start').onclick = startInspection;
$('measure').onclick = () => {
  if (busy) return;
  busy = true; controls();
  captureAndMeasure().finally(() => { busy = false; controls(); });
};
document.addEventListener('visibilitychange', () => { if (document.hidden && busy) abortInspection('Inspection paused because the page is hidden. Press Start inspection again.'); });
window.addEventListener('pagehide', () => { abortInspection(); stream?.getTracks().forEach(track => track.stop()); });
video.addEventListener('resize', () => {
  if (cameraReady && (video.videoWidth !== overlay.width || video.videoHeight !== overlay.height)) resetSetup('Camera resolution changed. Recalibrate before continuing.');
});
async function loadReferences() {
  try {
    const response = await fetch('/size-references');
    if (!response.ok) throw new Error('Reference guide could not be loaded.');
    const data = await response.json(), names = ['front rise', 'full length', 'hips', 'waist'];
    $('tolerance').textContent = data.tolerance_in.toFixed(2);
    const header = document.createElement('tr');
    for (const name of ['Size', ...names]) { const th = document.createElement('th'); th.textContent = name; header.append(th); }
    $('references').append(header);
    for (const size of ['M', 'L', 'XL', '2XL']) {
      const dimensions = data.references[size];
      if (!dimensions) continue;
      const row = document.createElement('tr');
      for (const text of [size, ...names.map(name => dimensions[name].toFixed(2))]) { const td = document.createElement('td'); td.textContent = text; row.append(td); }
      $('references').append(row);
    }
  } catch (error) { $('references').textContent = error.message; }
}
void loadReferences();
void startCamera();
