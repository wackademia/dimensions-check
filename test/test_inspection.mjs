import assert from 'node:assert/strict';
import {test} from 'node:test';
import {InspectionGate, changedFraction} from '../phone_web/inspection.mjs';

const frame = value => new Uint8ClampedArray(400).fill(value);
const background = frame(0), item = frame(120), different = frame(220);
function capture(gate, from = 0, sample = item) {
  gate.update(sample, background, from);
  gate.update(sample, background, from + 250);
  return gate.update(sample, background, from + 1500);
}
test('empty belt never triggers', () => {
  const gate = new InspectionGate();
  for (let time = 0; time < 5000; time += 250) assert.equal(gate.update(background, background, time).state, 'empty');
});
test('paused item triggers once and accepts its result', () => {
  const gate = new InspectionGate();
  const result = capture(gate);
  assert.equal(result.state, 'capture'); assert.ok(gate.accepts(result.token));
  assert.equal(gate.update(item, background, 5000).state, 'holding');
});
test('moving item never triggers', () => {
  const gate = new InspectionGate();
  for (let t = 0; t < 5000; t += 250) assert.equal(gate.update(t % 500 ? item : different, background, t).state, 'moving');
});
test('movement invalidates a pending or displayed result', () => {
  const gate = new InspectionGate(), result = capture(gate);
  assert.equal(gate.update(different, background, 2000).state, 'clear');
  assert.ok(!gate.accepts(result.token));
  assert.equal(gate.update(different, background, 6000).state, 'clear');
});
test('empty gap rearms and rejects the old result for the next item', () => {
  const gate = new InspectionGate(), first = capture(gate);
  gate.update(background, background, 2000); gate.update(background, background, 2750);
  const second = capture(gate, 3000, different);
  assert.equal(second.state, 'capture'); assert.ok(gate.accepts(second.token));
  assert.ok(!gate.accepts(first.token));
});
test('brief empty flash is not enough to rearm', () => {
  const gate = new InspectionGate(); capture(gate);
  gate.update(background, background, 2000); gate.update(item, background, 2250);
  assert.equal(gate.update(item, background, 5000).state, 'clear');
});
test('manual reinspection invalidates the first result and waits for stability', () => {
  const gate = new InspectionGate(), first = capture(gate);
  gate.retry(); assert.ok(!gate.accepts(first.token));
  assert.equal(capture(gate, 2000).state, 'capture');
});
test('cumulative drift cannot pass the stability test', () => {
  const gate = new InspectionGate();
  gate.update(item, background, 0); gate.update(item, background, 250);
  for (let i = 1; i <= 6; i++) {
    const drifting = item.slice(); drifting.fill(240, 0, i * 4);
    assert.notEqual(gate.update(drifting, background, 250 + i * 250).state, 'capture');
  }
});
test('small pixel noise does not count as motion', () => {
  assert.equal(changedFraction(item, frame(125)), 0);
  assert.equal(changedFraction(item, background), 1);
});
