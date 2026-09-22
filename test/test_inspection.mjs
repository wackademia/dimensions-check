import assert from 'node:assert/strict';
import {test} from 'node:test';
import {SettleGate, changedFraction} from '../phone_web/inspection.mjs';

const frame = value => new Uint8ClampedArray(400).fill(value);
const item = frame(120), different = frame(220);

function settle(gate, from = 0, sample = item) {
  gate.update(sample, from);
  gate.update(sample, from + 250);
  return gate.update(sample, from + 1500);
}

test('a still item becomes ready after ~1.25s of stillness', () => {
  const gate = new SettleGate();
  assert.equal(settle(gate).state, 'ready');
});
test('moving item never becomes ready', () => {
  const gate = new SettleGate();
  for (let t = 0; t < 5000; t += 250) assert.equal(gate.update(t % 500 ? item : different, t).state, 'moving');
});
test('settling reports before settleMs of stillness has passed', () => {
  const gate = new SettleGate();
  gate.update(item, 0);
  assert.equal(gate.update(item, 250).state, 'settling');
});
test('movement resets the stability timer', () => {
  const gate = new SettleGate();
  gate.update(item, 0); gate.update(item, 250);
  assert.equal(gate.update(different, 500).state, 'moving');
  assert.equal(gate.update(item, 750).state, 'moving', 'still transitioning away from `different`');
  assert.equal(gate.update(item, 1000).state, 'settling', 'now stable again');
  assert.equal(gate.update(item, 2500).state, 'ready');
});
test('cumulative drift cannot pass the stability test', () => {
  const gate = new SettleGate();
  gate.update(item, 0); gate.update(item, 250);
  for (let i = 1; i <= 6; i++) {
    const drifting = item.slice(); drifting.fill(240, 0, i * 4);
    assert.notEqual(gate.update(drifting, 250 + i * 250).state, 'ready');
  }
});
test('a fresh gate has no presence/empty concept: a still empty scene also becomes ready', () => {
  // Unlike the old InspectionGate, SettleGate only waits for stillness; whether an item
  // is actually present is left to the server's own frame-quality checks after capture.
  const gate = new SettleGate();
  assert.equal(settle(gate, 0, frame(0)).state, 'ready');
});
test('small pixel noise does not count as motion', () => {
  assert.equal(changedFraction(item, frame(125)), 0);
  assert.equal(changedFraction(item, frame(0)), 1);
});
