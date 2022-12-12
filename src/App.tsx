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
  min: number;
  max: number;
  step?: "any" | number;
};

const SliderWithName: React.FC<SliderWithNameProp> = (props: SliderWithNameProp) => {
  return (
    <S.SliderWithName>
      <div className="slider-name">{props.name}</div>
      <S.SliderBody type="range" defaultValue={props.defaultValue} min={props.min} max={props.max} step={props.step || "any"}
  onChange={(event) => props.onChangeParameter(props.parameterId, parseFloat(event.target.value))}>
      </S.SliderBody>
    </S.SliderWithName>
  );
}

const Control: React.FC<{ onChangeParameter: (parameterId: string, value: number) => void }> = (props) => {
  return (
    <S.ParameterListLayout>
      <SliderWithName
        name={"Dry Wet"}
        parameterId={"dry-wet"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0.5}
        min={0.0}
        max={1.0}
      />
      <SliderWithName
        name={"Output Gain"}
        parameterId={"output-gain"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0}
        min={-48}
        max={6}
      />
      <SliderWithName
        name={"Pitch Shift"}
        parameterId={"pitch-shift"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0}
        min={-1200}
        max={1200}
        step={1}
      />
      <SliderWithName
        name={"Formant Shift"}
        parameterId={"formant-shift"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={0}
        min={-1}
        max={1}
      />
       <SliderWithName
        name={"Envelope Order"}
        parameterId={"envelope-order"}
        onChangeParameter={props.onChangeParameter}
        defaultValue={7}
        min={3}
        max={30}
        step={1}
      />
    </S.ParameterListLayout>
  );
}

function MainPanel() {
  const [audioManager, setAudioManager] = useState<A.AudioManager|undefined>(undefined);
  const [deviceList, setDeviceList] = useState<A.DeviceList | undefined>(undefined);
  const [inputDevice, setInputDevice] = useState<string>("");
  const [running, setRunning] = useState<boolean>(false);
  const [levels, setLevels] = useState<A.LevelMeterState>({inputLevel: 0, outputLevel: 0});

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
    setLevels(levels);
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
    audioManager.node.port.postMessage({
      type: "set-" + type,
      value
    });   
  };

  // Audio already initialized. Suspend / resume based on its current state.
  const { context } = audioManager;
  return (
    <S.MainPanelLayout>
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
      <S.AudioPanelLayout>
        <LevelMeter level={levels.inputLevel} maxDecibel={6} minDecibel={-48} />
        <Control onChangeParameter={changeParameter} />
        <LevelMeter level={levels.outputLevel} maxDecibel={6} minDecibel={-48} />
      </S.AudioPanelLayout>
    </S.MainPanelLayout>
  );
}

function App() {
  return (
    <S.App>
      <S.AppHeader>
        Wasm Audio Voice Changer
      </S.AppHeader>
      <S.AppContent>
        <MainPanel />
      </S.AppContent>
    </S.App>
  );
}

export default App;
