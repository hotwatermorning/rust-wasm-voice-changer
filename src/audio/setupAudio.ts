import ProcessorNode from "./ProcessorNode";
import { LevelMeterState } from "./LevelMeter";

type ProcessCallbackFunction = (levels: LevelMeterState) => void;

export type AudioManager = {
  context: AudioContext;
  node: ProcessorNode;
};

export async function setupAudio(deviceId: string, onProcessCallback: ProcessCallbackFunction): Promise<AudioManager> {
  // Get the browser audio. Awaits user "allowing" it for the current tab.
  const mediaStream = await window.navigator.mediaDevices.getUserMedia({
    audio: { deviceId: deviceId },
    video: false
  });

  const context = new window.AudioContext({
    latencyHint: "balanced",
    sampleRate: 44100
  });
  const audioSource = context.createMediaStreamSource(mediaStream);

  let node;

  try {
    // Fetch the WebAssembly module that performs processing audio.
    const response = await window.fetch("wasm-audio/wasm_audio_bg.wasm");
    const wasmBytes = await response.arrayBuffer();

    // Add our audio processor worklet to the context.
    const processorUrl = "Processor.js";
    try {
      await context.audioWorklet.addModule(processorUrl);
    } catch (e) {
      let err = e as unknown as Error;
      throw new Error(
        `Failed to load audio analyzer worklet at url: ${processorUrl}. Further info: ${err.message}`
      );
    }

    // Create the AudioWorkletNode which enables the main JavaScript thread to
    // communicate with the audio processor (which runs in a Worklet).
    node = new ProcessorNode(context, "Processor");

    // blockSize specifies the number of consecutive audio samples of
    // each processing unit of work. Larger values tend
    // to produce slightly more accurate results but are more expensive to compute and
    // can lead to notes being missed in faster passages i.e. where the music note is
    // changing rapidly. 1024 is usually a good balance between efficiency and accuracy
    // for music analysis.
    const blockSize = 128;

    // Send the Wasm module to the audio node which in turn passes it to the
    // processor running in the Worklet thread. Also, pass any configuration
    // parameters for the Wasm audio processing.
    node.init(wasmBytes, onProcessCallback, blockSize, 0.2, 0.5, 0.5);

    // Connect the audio source (microphone output) to our analysis node.
    audioSource.connect(node);

    // Connect our analysis node to the output. Required even though we do not
    // output any audio. Allows further downstream audio processing or output to
    // occur.
    node.connect(context.destination);
  } catch (e) {
    let err = e as unknown as Error;
    throw new Error(
      `Failed to load audio analyzer WASM module. Further info: ${err.message}`
    );
  }

  return { context, node };
}
