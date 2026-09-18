/**
 * Runs the browser client's audio-thread code outside a browser.
 *
 *   node capture_check.js <fusion-runtime.js> <hardware rate> <target rate>
 *
 * Feeds it one second of a 440 Hz tone and prints what came out, so the test
 * in test_web.py can check the microphone really is resampled to the rate the
 * server asked for. Getting this wrong doesn't crash anything — it just feeds
 * Whisper audio at the wrong speed, which sounds like nonsense.
 */
const fs = require("fs");

const [, , clientPath, hardwareRate, targetRate] = process.argv;
const chunkMs = 40;
const messages = [];

global.window = global;
global.document = { currentScript: null };
global.location = { origin: "http://localhost:8000" };
(0, eval)(fs.readFileSync(clientPath, "utf8"));

let registered = null;
global.registerProcessor = (name, cls) => { registered = cls; };
global.AudioWorkletProcessor = class {
  constructor() {
    this.port = { postMessage: (message) => messages.push(message) };
  }
};
global.sampleRate = Number(hardwareRate);
global.FusionRuntime.captureProcessor();

const processor = new registered({ processorOptions: { targetRate: Number(targetRate), chunkMs } });
const blockSize = 128;
for (let block = 0; block < global.sampleRate / blockSize; block++) {
  const samples = new Float32Array(blockSize);
  for (let i = 0; i < blockSize; i++) {
    samples[i] = Math.sin(2 * Math.PI * 440 * ((block * blockSize + i) / global.sampleRate)) * 0.5;
  }
  processor.process([[samples]]);
}

const samples = messages.reduce((total, m) => total + m.pcm.byteLength / 2, 0);
console.log(JSON.stringify({
  chunks: messages.length,
  samples: samples,
  chunk_samples: messages.length ? messages[0].pcm.byteLength / 2 : 0,
  level: messages.length ? messages[messages.length - 1].level : 0,
}));
