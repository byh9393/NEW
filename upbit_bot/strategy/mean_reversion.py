"""
Mean reversion plugin using Bollinger Bands Z-score and RSI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import bollinger_bands, rsi
from upbit_bot.strategy.composite import Decision, Signal


@dataclass
class MeanRevConfig:
    z_entry: float = 2.0
    rsi_low: float = 30
    rsi_high: float = 70
    min_score: float = 10.0


def mean_reversion_signal(market: str, prices: pd.Series, config: MeanRevConfig = MeanRevConfig()) -> Decision:
    if prices.size < 40:
        return Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 부족")

    bb = bollinger_bands(prices)
    upper = bb["upper"].iloc[-1]
    lower = bb["lower"].iloc[-1]
    mid = bb["middle"].iloc[-1]
    last = float(prices.iloc[-1])
    z = (last - mid) / (upper - lower + 1e-9)
    rsi_val = float(rsi(prices).iloc[-1])

    if z < -config.z_entry and rsi_val < config.rsi_low:
        score = max(config.min_score, abs(z) * 20)
        return Decision(market, last, score, Signal.BUY, f"BB 과매도 Z={z:.2f}, RSI={rsi_val:.1f}", strategy="mean_reversion")
    if z > config.z_entry and rsi_val > config.rsi_high:
        score = -max(config.min_score, abs(z) * 20)
        return Decision(market, last, score, Signal.SELL, f"BB 과매수 Z={z:.2f}, RSI={rsi_val:.1f}", strategy="mean_reversion")

    return Decision(market, last, 0.0, Signal.HOLD, "평균회귀 없음", strategy="mean_reversion")
