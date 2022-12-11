import React, { useEffect, useState } from "react";
import * as A from "./audio";
import * as S from "./styles";
import { DeviceSelector, LevelMeter } from "./presentation/components";
import "./App.css";

type SliderWithNameProp = {
  name: string;
  parameterId: string;
  onChangeParameter: (parameterId: string, value: number) => void;
  defaultValue: number;
};

const SliderWithName: React.FC<SliderWithNameProp> = (props: SliderWithNameProp) => {
  return (
    <div className="slider-with-name">
      <div className="slider-name">{props.name}</div>
      <input className="slider-body" type="range" defaultValue={props.defaultValue} min={0} max={1} step={"any"}
  onChange={(event) => props.onChangeParameter(props.parameterId, parseFloat(event.target.value))}>
      </input>
    </div>
  );
}

const Control: React.FC<{ onChangeParameter: (parameterId: string, value: number) => void }> = (props) => {
  return (
    <>
      <SliderWithName
        name={"Wet Amount"}
        parameterId={"wet-amount"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0.5}
      />
      <SliderWithName
        name={"Feedback"}
        parameterId={"feedback-amount"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0.5}
      />
      <SliderWithName
        name={"Delay Length"}
        parameterId={"delay-length"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0.2}
      />
    </>
  );
}

function AudioControl() {
  const [audioManager, setAudioManager] = useState<A.AudioManager|undefined>(undefined);
  const [deviceList, setDeviceList] = useState<A.DeviceList | undefined>(undefined);
  const [inputDevice, setInputDevice] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      setDeviceList(await A.enumerateDevices());
    })();
  }, [setDeviceList]);

  if(deviceList === undefined || deviceList.inputDevices.length === 0) {
    return (
      <div>使用できるオーディオ入力デバイスがありません。</div>
    );
  }

  if(inputDevice === "") {
    return (
      <div id={"device-selector-layout"}>
        <DeviceSelector devices={deviceList} onSelected={(deviceId: string) => {
          setInputDevice(deviceId);
        }} />
      </div>
    );
  }

  const callback = (levels: A.LevelMeterState) => {
  };

  const onStartListening = async () => {
    setAudioManager(await A.setupAudio(inputDevice, callback));
    setRunning(true);
  };

  // Initial state. Initialize the web audio once a user gesture on the page
  // has been registered.
  if (!audioManager) {
    return (
      <button onClick={onStartListening}>
        Start listening
      </button>
    );
  }

  const changeParameter = (type: string, value: number) => {
    if(type === "wet-amount") {
      audioManager.node.port.postMessage({
        type: "set-wet-amount",
        value
      });
    } else if(type === "feedback-amount") {
      audioManager.node.port.postMessage({
        type: "set-feedback-amount",
        value
      });
    } else if(type === "delay-length") {
      audioManager.node.port.postMessage({
        type: "set-delay-length",
        value
      });
    }
  };

  // Audio already initialized. Suspend / resume based on its current state.
  const { context } = audioManager;
  return (
    <div>
      <button
        onClick={async () => {
          if (running) {
            await context.suspend();
            setRunning(context.state === "running");
          } else {
            await context.resume();
            setRunning(context.state === "running");
          }
        }}
        disabled={context.state !== "running" && context.state !== "suspended"}
      >
        {running ? "Pause" : "Resume"}
      </button>
      <Control onChangeParameter={changeParameter} />
    </div>
  );
}

function App() {
  return (
    <S.App>
      <S.AppHeader>
        Wasm Audio Voice Changer
      </S.AppHeader>
      <S.AppContent>
        <AudioControl />
      </S.AppContent>
    </S.App>
  );
}

export default App;