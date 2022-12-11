import { FormControl, MenuItem } from "@mui/material";
import { DeviceList, DeviceInfo } from "audio";
import * as S from "./styles";

type Props = {
  devices: DeviceList;
  onSelected: (deviceId: string) => void;
};

export const DeviceSelector: React.FC<Props> = ({devices, onSelected}) => {
  const inputDevices = devices.inputDevices;

  return (
    <S.DeviceSelector>
      <FormControl fullWidth variant="filled" sx={{ m: 1, minWidth: 120 }}>
        <S.DeviceInputLabel id="select-input-device">Select Input Device</S.DeviceInputLabel>
        <S.DeviceSelect
          labelId="select-input-device"
          id="inputDevice"
          label="InputDevice"
          value={""}
          onChange={(event) => onSelected(event.target.value as string)}
        >
          <MenuItem value={""} disabled><em>None</em></MenuItem>
          {inputDevices.map((info: DeviceInfo) => <MenuItem key={info.deviceId} value={info.deviceId}>{info.label}</MenuItem>)}
        </S.DeviceSelect>
      </FormControl>
    </S.DeviceSelector>
  );
}