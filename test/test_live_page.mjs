// DOM/camera adapter tests. These are not a replacement for visual browser QA.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';
import {SettleGate} from '../phone_web/inspection.mjs';

const html = fs.readFileSync(new URL('../phone_web/index.html', import.meta.url), 'utf8');
const code = fs.readFileSync(new URL('../phone_web/app.js', import.meta.url), 'utf8')
  .replace("import {SettleGate} from './inspection.mjs';", '');
const flush = () => new Promise(resolve => setImmediate(resolve));

async function page() {
  let now = 0, colour = 0, interval;
  const pending = [], requests = [], events = {};
  class Element {
    constructor() { this.value = ''; this.textContent = ''; this.disabled = false; this.hidden = false; this.children = []; this.events = {}; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    removeAttribute(key) { delete this[key]; }
    addEventListener(type, callback) { this.events[type] = callback; }
    getBoundingClientRect() { return {left: 0, top: 0, width: 1280, height: 960}; }
    getContext() { return {
      clearRect() {}, strokeRect() {}, drawImage() {}, beginPath() {}, arc() {}, fill() {},
      getImageData() { return {data: new Uint8ClampedArray(400).fill(colour)}; },
    }; }
    toBlob(callback) { callback({type: 'image/jpeg'}); }
  }
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  elements.mode.value = 'underwear';
  elements.resultPanel.hidden = true;
  const video = elements.video;
  Object.assign(video, {videoWidth: 1280, videoHeight: 960, readyState: 4, play: async () => {}});
  Object.defineProperty(video, 'currentTime', {get: () => now / 1000});
  const track = {stop() {}, getSettings: () => ({deviceId: 'test-camera'})};
  const document = {
    getElementById: id => elements[id], createElement: () => new Element(),
    addEventListener: (type, callback) => { events[type] = callback; }, hidden: false,
  };
  class Form { constructor() { this.fields = new Map(); } append(...args) { this.fields.set(args[0], args[1]); } }
  vm.runInNewContext(code, {
    SettleGate, document, window: {addEventListener() {}},
    navigator: {mediaDevices: {getUserMedia: async () => ({getTracks: () => [track], getVideoTracks: () => [track]}),
      enumerateDevices: async () => [{kind: 'videoinput', deviceId: 'test-camera', label: 'Overhead test camera'}]}},
    Option: function (label, value) { return {label, value}; },
    performance: {now: () => now}, Date, FormData: Form, AbortController,
    setInterval: callback => { interval = callback; return 1; }, clearInterval: () => { interval = null; },
    setTimeout: () => 1, clearTimeout() {},
    fetch: async (url, options) => {
      if (url === '/size-references') return {ok: true, json: async () => ({tolerance_in: .5, references: {L: {'front rise': 11.14, 'full length': 12.65, hips: 16.72, waist: 14.62}}})};
      requests.push(options);
      return new Promise(resolve => pending.push(resolve));
    },
  });
  await flush();
  const api = {
    elements, document, events, requests,
    async tick(value, count = 1) { colour = value; for (let i = 0; i < count; i++) { now += 250; interval?.(); await flush(); } },
    async setup() {
      elements.calibrate.onclick();
      elements.overlay.events.pointerdown({clientX: 100, clientY: 100});
      elements.overlay.events.pointerdown({clientX: 700, clientY: 100});
      await flush();
    },
    async reply(ok = true) {
      assert.ok(pending.length, 'a request must be pending');
      pending.shift()({ok, status: ok ? 200 : 429, json: async () => ok ? {
        garment_class: 'shorts', inference_seconds: 2, image: 'data:image/jpeg;base64,test',
        measurements: {waist: {distance: 14.62, unit: 'in'}},
        size_estimate: {status: 'matched', size: 'L', reason: 'Reference match', reference: {waist: 14.62}},
      } : {error: 'Model busy'}});
      await flush();
    },
  };
  return api;
}

test('page requires calibration; frozen selection works', async () => {
  const p = await page();
  assert.ok(p.elements.start.disabled);
  assert.equal(p.elements.tolerance.textContent, '0.50');
  assert.equal(p.elements.references.children.length, 2);
  await p.setup();
  assert.ok(!p.elements.start.disabled);
  assert.match(p.elements.setup.textContent, /50.00 pixels\/inch/);
  p.elements.area.onclick();
  p.elements.overlay.events.pointerdown({clientX: 10, clientY: 10});
  p.elements.overlay.events.pointerdown({clientX: 1200, clientY: 900});
  assert.match(p.elements.setup.textContent, /50.00 pixels\/inch/, 'crop preserves pixel scale');
  assert.ok(!p.elements.start.disabled, 'no empty-area capture step is required anymore');
});
test('one press: waits for stillness, measures once, then re-enables the button', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick();
  assert.ok(p.elements.start.disabled, 'disabled while waiting/measuring');
  await p.tick(120, 8);
  assert.equal(p.requests.length, 1);
  assert.equal(p.requests[0].body.fields.get('mode'), 'underwear');
  assert.equal(p.requests[0].body.fields.get('pixels_per_inch'), 50);
  await p.reply();
  assert.equal(p.elements.size.textContent, 'Estimated L');
  assert.equal(p.elements.resultPanel.hidden, false);
  assert.ok(!p.elements.start.disabled, 'reengageable once the result is shown');
});
test('the result stays on screen until the next inspection, regardless of motion', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick();
  await p.tick(120, 8); await p.reply();
  assert.equal(p.elements.size.textContent, 'Estimated L');
  await p.tick(220, 20); // item removed/replaced; nothing polls or clears automatically
  assert.equal(p.elements.size.textContent, 'Estimated L');
  assert.equal(p.elements.resultPanel.hidden, false);
});
test('a second press measures the next item and overwrites the previous result', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick();
  await p.tick(120, 8); await p.reply();
  p.elements.start.onclick();
  assert.ok(p.elements.start.disabled, 'measuring the next item');
  await p.tick(120, 8);
  assert.equal(p.requests.length, 2);
  await p.reply();
  assert.equal(p.elements.size.textContent, 'Estimated L');
});
test('late result is discarded when setup is invalidated mid-wait', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick();
  await p.tick(120, 4);
  p.elements.video.videoWidth = 640; p.elements.video.events.resize();
  assert.ok(p.elements.start.disabled, 'recalibration required');
  assert.match(p.elements.status.textContent, /resolution changed/);
});
test('server failures allow an immediate retry', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick();
  await p.tick(120, 8); await p.reply(false);
  assert.equal(p.elements.size.textContent, 'Inspection failed');
  assert.ok(!p.elements.start.disabled);
  p.elements.start.onclick(); await p.tick(120, 8);
  assert.equal(p.requests.length, 2);
});
test('camera resolution change invalidates calibration and pending results', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick(); await p.tick(120, 8);
  p.elements.video.videoWidth = 640; p.elements.video.events.resize(); await p.reply();
  assert.ok(p.elements.start.disabled); assert.ok(p.elements.resultPanel.hidden);
  assert.match(p.elements.status.textContent, /resolution changed/);
});
test('hidden tab aborts the pending settle wait and ignores its late response', async () => {
  const p = await page(); await p.setup(); p.elements.start.onclick(); await p.tick(120, 2);
  p.document.hidden = true; p.events.visibilitychange();
  assert.equal(p.elements.size.textContent, 'Stopped');
  assert.ok(!p.elements.start.disabled, 'nothing was sent to the server yet, so it re-enables immediately');
});
