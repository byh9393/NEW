"""
멀티 타임프레임 지표 뷰어 및 팩터 엔진.

- 5m/15m/1h/4h/1d 종가·거래량을 동시에 읽어 트렌드/모멘텀/변동성/거래량/상관성 점수를 0~1로 정규화
- 팩터별 가중치를 적용해 하나의 합성 스코어를 생성해 진입 필터로 활용
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import ema, supertrend


def _normalize(value: float, low: float, high: float) -> float:
    if high - low <= 1e-9:
        return 0.5
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


@dataclass
class MultiTimeframeFactor:
    trend: float
    momentum: float
    volatility: float
    volume: float
    correlation: float
    composite: float
    reasons: List[str]
    trend_by_tf: Dict[str, float]
    regime: float


class MultiTimeframeAnalyzer:
    def __init__(self, timeframes: Iterable[str]) -> None:
        self.timeframes = tuple(timeframes)
        self.factor_weights = {
            "trend": 0.7,
            "momentum": 0.15,
            "volatility": 0.05,
            "volume": 0.05,
            "correlation": 0.05,
        }

    def analyze(
        self,
        market: str,
        frames: Dict[str, pd.DataFrame],
        *,
        correlation: Optional[float] = None,
    ) -> MultiTimeframeFactor:
        trend_scores: Dict[str, float] = {}
        momentum_scores: List[float] = []
        volatility_scores: List[float] = []
        volume_scores: List[float] = []
        reasons: List[str] = []

        for tf in self.timeframes:
            frame = frames.get(tf)
            if frame is None or frame.empty or len(frame) < 20:
                continue
            closes = frame["close"]
            price = closes.iloc[-1]
            st = supertrend(closes)
            st_dir = int(st["direction"].iloc[-1]) if not st.empty else 0
            ema_long = ema(closes, 200).iloc[-1] if len(closes) >= 50 else price
            trend_norm = 1.0 if (price > ema_long and st_dir == 1) else 0.0 if (price < ema_long and st_dir == -1) else 0.5
            trend_scores[tf] = trend_norm

            momentum_scores.append(trend_norm)
            volatility_scores.append(0.5)

            if "volume" in frame.columns:
                volume_scores.append(0.5)

            reasons.append(
                f"{market} {tf} Supertrend 방향 {st_dir}, 200EMA 대비 {'상' if price > ema_long else '하' if price < ema_long else '동일'}"
            )

        trend = float(np.clip(np.mean(list(trend_scores.values()) or [0.5]), 0.0, 1.0))
        momentum = float(np.clip(np.mean(momentum_scores or [0.5]), 0.0, 1.0))
        volatility = float(np.clip(np.mean(volatility_scores or [0.5]), 0.0, 1.0))
        volume = float(np.clip(np.mean(volume_scores or [0.5]), 0.0, 1.0))
        corr_factor = 1.0 - abs(correlation) if correlation is not None else 0.5
        corr_factor = float(np.clip(corr_factor, 0.0, 1.0))
        regime = float(np.clip((trend_scores.get("1d", 0.5) + trend_scores.get("4h", 0.5)) / 2, 0.0, 1.0))

        composite = (
            trend * self.factor_weights["trend"]
            + momentum * self.factor_weights["momentum"]
            + volatility * self.factor_weights["volatility"]
            + volume * self.factor_weights["volume"]
            + corr_factor * self.factor_weights["correlation"]
        )

        return MultiTimeframeFactor(
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            volume=volume,
            correlation=corr_factor,
            composite=float(np.clip(composite, 0.0, 1.0)),
            reasons=reasons,
            trend_by_tf=trend_scores,
            regime=regime,
        )
