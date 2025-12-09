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

from upbit_bot.indicators.technical import bollinger_bands, ema, rsi, atr_like


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
            "trend": 0.3,
            "momentum": 0.25,
            "volatility": 0.15,
            "volume": 0.2,
            "correlation": 0.1,
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
            atr = atr_like(closes).iloc[-1]
            price = closes.iloc[-1]

            fast = ema(closes, 8).iloc[-1]
            slow = ema(closes, 34).iloc[-1]
            slope = (fast - slow) / (price + 1e-9)
            trend_norm = _normalize(np.tanh(slope * 12), -1.0, 1.0)
            trend_scores[tf] = trend_norm

            rsi_val = rsi(closes).iloc[-1]
            momentum_scores.append(_normalize(rsi_val, 35, 70))

            bb = bollinger_bands(closes)
            width = (bb["upper"].iloc[-1] - bb["lower"].iloc[-1]) / (price + 1e-9)
            atr_ratio = atr / (price + 1e-9)
            vol_signal = _normalize(0.5 - abs(atr_ratio - 0.01) * 12, 0.0, 1.0)
            squeeze_signal = _normalize(width, 0.01, 0.08)
            volatility_scores.append((vol_signal + squeeze_signal) / 2)

            if "volume" in frame.columns:
                recent_vol = frame["volume"].iloc[-1]
                median_vol = frame["volume"].rolling(window=20).median().iloc[-1]
                volume_scores.append(_normalize(recent_vol / (median_vol + 1e-9), 0.5, 2.0))

            reasons.append(
                f"{market} {tf} 트렌드 {trend_norm:.2f}, RSI {rsi_val:.1f}, 밴드폭 {width:.3f}, ATR {atr_ratio:.3f}"
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
