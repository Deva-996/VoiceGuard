// AudioWorkletProcessor: buffers mic/remote audio into ~85 ms Int16 frames and posts them
// to the main thread. Runs off the main thread (unlike the deprecated ScriptProcessorNode).
// Downsampling to 16 kHz is left to the backend (it gets input_sample_rate over the WS).

class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const frameMs = (options.processorOptions && options.processorOptions.frameMs) || 85;
    this._frameLen = Math.round((sampleRate * frameMs) / 1000);
    this._buf = new Float32Array(this._frameLen);
    this._n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0]; // mono (first channel)
    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === this._frameLen) {
        const i16 = new Int16Array(this._frameLen);
        for (let j = 0; j < this._frameLen; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          i16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(i16.buffer, [i16.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
