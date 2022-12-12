import { keyframes } from "styled-components";
import { styled } from "@mui/material/styles";

export const App = styled("div")`
  text-align: center;
  background-color: #282c34;
  color: white;
  height: 100vh;
`;

export const AppLogo = styled("div")`
  height: 40vmin;
  pointer-events: none;

  @media (prefers-reduced-motion: no-preference) {
    animation: App-logo-spin infinite 20s linear;
  }
`;

export const AppHeader = styled("header")`
  height: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: calc(10px + 2vmin);
`;

export const AppLink = styled("div")`
  color: #61dafb;
`;

export const AppLogoSpin = keyframes`
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
`;

export const AppContent = styled("div")`
`;

export const SliderWithName = styled("div")`
  padding: 20px;
`;

export const SliderBody = styled("input")`
  width: 200px;
`;

export const AudioPanelLayout = styled("div")`
  display: flex;
  gap: 20px;
`;

export const ParameterListLayout = styled("div")`
`;