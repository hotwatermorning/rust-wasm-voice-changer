export type DeviceInfo = {
  label: string;
  deviceId: string;
};

export type DeviceList = {
  inputDevices: DeviceInfo[];
  outputDevices: DeviceInfo[];
};

export const enumerateDevices = async (): Promise<DeviceList> => {
  if (!window.navigator.mediaDevices) {
    throw new Error(
      "This browser does not support web audio or it is not enabled."
    );
  }

  let devices: DeviceList = {
    inputDevices: [],
    outputDevices: [],
  };

  const foundDevices = await navigator.mediaDevices.enumerateDevices();

  for(let found of foundDevices) {
    if(found.kind === "audioinput") {
      devices.inputDevices.push(found);
    } else if(found.kind === "audiooutput") {
      devices.outputDevices.push(found);
    }
  }

  return devices;
};