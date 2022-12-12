export type LevelMeterState = {
  inputLevel: number;
  outputLevel: number;
};

export type ProcessCallbackType = (level: LevelMeterState) => void;

// function isLevelMeterState(arg: any): arg is LevelMeterState {
//   return \
//   arg.inputLevel !== undefined &&
//   arg.outputLevel !== undefined &&
//   typeof(arg.inputLevel) ===  &&
//   arg.outputLevel !== undefined &&
// }

export default class ProcessorNode extends AudioWorkletNode {
  private onProcessCallback: ProcessCallbackType = (level: LevelMeterState) => {};
  private blockSize: number = 0;
  private delayLength: number = 0;
  private wetAmount: number = 0;
  private feedback: number = 0;

  /**
   * Initialize the Audio processor by sending the fetched WebAssembly module to
   * the processor worklet.
   *
   * @param {ArrayBuffer} wasmBytes Sequence of bytes.
   * @param {number} blockSize Number of audio samples used
   * for each analysis. Must be a power of 2.
   */
  init(
    wasmBytes: ArrayBuffer,
    onProcessCallback: ProcessCallbackType,
    blockSize: number,
    delayLength: number,
    wetAmount: number,
    feedback: number
  ) {
    this.onProcessCallback = onProcessCallback;
    this.blockSize = blockSize;
    this.delayLength = delayLength;
    this.wetAmount = wetAmount;
    this.feedback = feedback;

    // Listen to messages sent from the audio processor.
    this.port.onmessage = (event: MessageEvent) => {
      this.onmessage(event.data);
    }

    this.port.postMessage({
      type: "send-wasm-module",
      wasmBytes,
    });

    // Handle an uncaught exception thrown in the Processor.
    this.onprocessorerror = (err) => {
      console.log(
        `An error from AudioWorkletProcessor.process() occurred: ${err}`
      );
    };
  }

  onmessage = (event: MessageEvent) => {
    if (event.type === 'wasm-module-loaded') {
      // The Wasm module was successfully sent to the Processor running on the
      // AudioWorklet thread and compiled. This is our cue to process audio.
      this.port.postMessage({
        type: "init-processor",
        sampleRate: this.context.sampleRate,
        blockSize: this.blockSize,
        delayLength: this.delayLength,
        wetAmount: this.wetAmount,
        feedback: this.feedback,
      });
    } else if (event.type === "update-levels") {
      // Receive level values. Invoke our callback which will result in the UI updating.
      this.onProcessCallback({
        inputLevel: event.data.inputLevel,
        outputLevel: event.data.outputLevel
      } as LevelMeterState);
    } else if (
      event.type === "set-dry-wet" ||
      event.type === "set-output-gain" ||
      event.type === "set-pitch-shift" ||
      event.type === "set-formant-shift" ||
      event.type === "set-envelope-order"
    ) {
      this.port.postMessage({
        type: event.type,
        value: event.data.value
      });
    }
  }
}
