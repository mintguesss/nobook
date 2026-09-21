/**
 * AudioWorklet 錄音處理器（規格 §4.1）。
 *
 * 刻意不用已棄用的 ScriptProcessorNode。AudioContext 以 sampleRate: 16000
 * 建立，瀏覽器會替我們降頻，這裡只負責把 128-sample 的區塊累積成
 * 250ms（4000 sample）的 PCM16 封包丟回主執行緒。
 */
const CHUNK_SAMPLES = 4000; // 250ms @ 16kHz

class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(CHUNK_SAMPLES);
    this.filled = 0;
    this.muted = false;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'mute') this.muted = !!e.data.value;
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch) return true;

    for (let i = 0; i < ch.length; i++) {
      this.buf[this.filled++] = ch[i];
      if (this.filled === CHUNK_SAMPLES) {
        if (!this.muted) this.flush();
        this.filled = 0;
      }
    }
    return true;
  }

  flush() {
    // float32 → PCM16 little-endian
    const pcm = new Int16Array(CHUNK_SAMPLES);
    for (let i = 0; i < CHUNK_SAMPLES; i++) {
      const s = Math.max(-1, Math.min(1, this.buf[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    this.port.postMessage(pcm.buffer, [pcm.buffer]);
  }
}

registerProcessor('recorder', RecorderProcessor);
