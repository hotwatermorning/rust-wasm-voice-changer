import { styled } from "@mui/material/styles";
import { InputLabel, Select } from "@mui/material";

export const DeviceSelector = styled("div")`
  width: 400px;
  margin: auto;
`;

export const DeviceInputLabel = styled(InputLabel)`
  color: white;
`;

export const DeviceSelect = styled(Select)`
  border-color: "yellow";
  background-color: "red";

  border-color: purple;
  color: magenta;

  & .MuiOutlinedInput-notchedOutline {
    border-color: lightgray;
    color: white;
  }

  &:hover .MuiOutlinedInput-notchedOutline {
    border-color: cyan;
  }
`;