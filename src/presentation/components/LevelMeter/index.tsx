import React from "react";
import * as S from "./styles";

type Props = {
  decibel: number;
  minDecibel: number;
  maxDecibel: number;
};

export const LevelMeter: React.FC<Props> = (props) => {
  props.decibel = Math.min(props.maxDecibel, Math.max(props.decibel, props.minDecibel));
  let range = props.maxDecibel - props.minDecibel;
  let ratio = (props.decibel - props.minDecibel) / range;

  let width = 40;
  let height = 200;
  let x = (1.0 - ratio) * height;
  return (
    <S.LevelMeter>
      <svg width={width} height={height} viewBox={`0, 0, ${width}, ${height}`} xmlns="http://www.w3.org/2000/svg">
        <rect x={0} y={0} width={width} height={height} fill="#241212"></rect>
        <rect x={x} y={0} width={width} height={height - x} fill="#7CFC00"></rect>
      </svg>
    </S.LevelMeter>
  );
};