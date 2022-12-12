import React from "react";
import * as S from "./styles";

type Props = {
  level: number;
  minDecibel: number;
  maxDecibel: number;
};

export const LevelMeter: React.FC<Props> = (props) => {
  let level = Math.min(props.maxDecibel, Math.max(props.level, props.minDecibel));
  let range = props.maxDecibel - props.minDecibel;
  let ratio = (level - props.minDecibel) / range;

  let width = 40;
  let height = 200;
  let y = (1.0 - ratio) * height;

  let color = level > 0 ? "#FF3322" : "#6CFC00";

  return (
    <S.LevelMeter>
      <svg width={width} height={height} viewBox={`0, 0, ${width}, ${height}`} xmlns="http://www.w3.org/2000/svg">
        <rect x={0} y={0} width={width} height={height} fill="#241212"></rect>
        <rect x={0} y={y} width={width} height={height - y} fill={color}></rect>
      </svg>
    </S.LevelMeter>
  );
};