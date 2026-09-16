import {InspectionGate} from './inspection.mjs';

const $ = id => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), context = overlay.getContext('2d');
const probe = document.createElement('canvas');
probe.width = 96; probe.height = 72;
const probeContext = probe.getContext('2d', {willReadFrequently: true});
let stream, roi, background, gate, timer, frozen;
let scale = 0, selection = null, points = [], running = false, busy = false, epoch = 0;
let cameraReady = false, lastState = '', lastVideoTime = -1, lastVideoAdvance = 0;
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
  for (const id of ['camera', 'mode', 'className', 'calibrate', 'area', 'background']) {
    $(id).disabled = busy || running || !!selection || !cameraReady;
  }
  $('background').disabled ||= !scale;
  $('start').disabled = !cameraReady || !scale || !background || busy || running || !!selection;
  $('stop').disabled = !running;
  $('retry').disabled = !running || busy;
  $('measure').disabled = !cameraReady || !scale || busy || !!selection;
  $('measure').hidden = isUnderwear();
  $('classControl').hidden = isUnderwear();
  $('start').hidden = $('stop').hidden = $('retry').hidden = !isUnderwear();
  $('background').hidden = !isUnderwear();
  $('cancel').hidden = !selection;
}
function drawArea() {
  context.clearRect(0, 0, overlay.width, overlay.height);
  if (!roi) return;
  context.strokeStyle = '#81e6b5'; context.lineWidth = 3;
  context.strokeRect(roi.x + 1, roi.y + 1, roi.w - 2, roi.h - 2);
}
function stop(message = 'Inspection stopped. No new frames will be submitted.') {
  running = false; epoch++; clearInterval(timer); timer = null; lastState = '';
  clearResult('Stopped'); say(message); controls();
}
function resetSetup(message) {
  stop(message); scale = 0; background = null; selection = null;
  $('setup').textContent = message;
  roi = {x: 0, y: 0, w: video.videoWidth, h: video.videoHeight};
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  drawArea(); controls();
}
async function startCamera(deviceId = '') {
  cameraReady = false;
  stop('Connecting to camera...');
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
    cameraReady = false; scale = 0; background = null;
    say(`Camera error: ${error.message}`); $('cameraStatus').textContent = 'No camera available. Check its connection and browser permission.';
    controls(); $('camera').disabled = false;
  }
}
function freezeFor(kind) {
  epoch++; background = null; selection = kind; points = []; clearResult('Setup');
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
  $('setup').textContent = scale ? `Calibrated: ${scale.toFixed(2)} pixels/inch. Area: ${roi.w} × ${roi.h} pixels. Empty-area reference needs capture.` : 'Calibrate the ruler before continuing.';
  say('Remove the ruler and any garment, then capture the empty inspection area. Do not move the camera.');
});

function renderResult(data, capturedAt, elapsed) {
  const decision = data.size_estimate;
  $('size').textContent = decision ? (decision.size ? `Estimated ${decision.size}` : 'Uncertain') : data.garment_class;
  $('size').className = `size ${decision?.status || ''}`;
  say(decision ? decision.reason : 'Measurement complete. Values below are in inches.');
  $('timing').textContent = `Inspected snapshot: ${capturedAt.toLocaleTimeString()} · model ${data.inference_seconds}s · round trip ${elapsed.toFixed(1)}s. This is not the current video frame.`;
  $('result').src = data.image; $('resultPanel').hidden = false;
  $('measurements').replaceChildren(...Object.entries(data.measurements).map(([name, value]) => {
    const row = document.createElement('tr');
    const reference = decision?.reference?.[name];
    for (const text of [name, `${value.distance.toFixed(2)} in`, reference == null ? '—' : `${reference.toFixed(2)} in`]) {
      const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
    }
    return row;
  }));
}
async function inspect(token = null) {
  if (busy) return;
  const requestEpoch = epoch, requestGate = gate, automatic = token !== null;
  const current = () => requestEpoch === epoch && (!automatic || (running && requestGate === gate && gate.accepts(token)));
  busy = true; controls(); clearResult('Inspecting'); say('Hold the item still until its result appears.');
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
      say(`${error.name === 'AbortError' ? 'Request timed out; the server may still be processing.' : error.message} Use Reinspect item to retry, or clear the area.`);
    }
  } finally {
    clearTimeout(timeout);
    busy = false; controls();
  }
}
function tick() {
  if (!running) return;
  if (!cameraReady || video.readyState < 2 || video.videoWidth !== overlay.width || video.videoHeight !== overlay.height) {
    resetSetup('Video changed or became unavailable. Recalibrate and capture the empty area again.'); return;
  }
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) { lastVideoTime = video.currentTime; lastVideoAdvance = now; }
  else if (now - lastVideoAdvance > 2000) { resetSetup('Camera feed froze. Check the camera, then repeat setup.'); return; }
  const state = gate.update(sample(), background, now);
  $('activity').textContent = `Area change ${(state.foreground * 100).toFixed(0)}% · motion ${(state.motion * 100).toFixed(1)}%${busy ? ' · model busy' : ''}`;
  if (state.state !== lastState) {
    const messages = {empty: 'Waiting for one item. Keep the inspection area empty between garments.',
      moving: 'Item is moving. Waiting for it to pause.', settling: 'Item detected. Waiting for a stable view.',
      clear: 'Item moved after inspection. Clear the area before the next item, or use Reinspect item.'};
    if (messages[state.state]) { clearResult(state.state === 'clear' ? 'Clear area' : 'Waiting'); say(messages[state.state]); }
    lastState = state.state;
  }
  if (state.state === 'capture') {
    if (busy) { gate.retry(); clearResult('Model busy'); say('Previous frame is still processing. Keep this item still.'); }
    else void inspect(state.token);
  }
}
$('camera').onchange = () => void startCamera($('camera').value);
$('calibrate').onclick = () => freezeFor('ruler');
$('area').onclick = () => freezeFor('area');
$('cancel').onclick = () => { selection = null; frozen = null; drawArea(); controls(); say('Selection cancelled. Capture the empty area again before starting.'); };
$('background').onclick = () => {
  background = sample(); epoch++; clearResult('Ready');
  $('setup').textContent = `${scale.toFixed(2)} pixels/inch · inspection area ${roi.w} × ${roi.h} · empty reference captured.`;
  say('Ready. Start inspection, then place one item flat with the waistband at the top.'); controls();
};
$('mode').onchange = () => { stop(); clearResult('Ready'); say(isUnderwear() ? 'Capture an empty area, then start automatic inspection.' : 'Select a garment type and measure a single frame.'); controls(); };
$('start').onclick = () => {
  running = true; epoch++; gate = new InspectionGate(); lastState = ''; lastVideoTime = -1; lastVideoAdvance = performance.now();
  clearResult(); controls(); timer = setInterval(tick, 250); tick();
};
$('stop').onclick = () => stop();
$('retry').onclick = () => { gate.retry(); lastState = ''; clearResult(); say('Waiting for this item to be still before another inspection.'); };
$('measure').onclick = () => void inspect();
document.addEventListener('visibilitychange', () => { if (document.hidden && (running || busy)) stop('Inspection paused because the page is hidden. Return here and press Start inspection.'); });
window.addEventListener('pagehide', () => { stop(); stream?.getTracks().forEach(track => track.stop()); });
video.addEventListener('resize', () => {
  if (cameraReady && (video.videoWidth !== overlay.width || video.videoHeight !== overlay.height)) resetSetup('Camera resolution changed. Recalibrate and capture the empty area again.');
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
