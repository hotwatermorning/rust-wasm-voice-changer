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
          onChange={(event) => onSelected(event.target.value as string)}
        >
          {inputDevices.map((info: DeviceInfo) => <MenuItem value={info.deviceId}>{info.label}</MenuItem>)}
        </S.DeviceSelect>
      </FormControl>
    </S.DeviceSelector>
  );
}