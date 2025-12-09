"""
Simple breakout strategy plugin using Donchian channels and volume confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from upbit_bot.strategy.composite import Decision, Signal


@dataclass
class BreakoutConfig:
    lookback: int = 50
    volume_mult: float = 1.5
    min_score: float = 10.0


def breakout_signal(market: str, prices: pd.Series, volumes: Optional[pd.Series], config: BreakoutConfig = BreakoutConfig()) -> Decision:
    if prices.size < config.lookback + 5:
        return Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 부족")

    recent_high = prices.tail(config.lookback).max()
    recent_low = prices.tail(config.lookback).min()
    last = float(prices.iloc[-1])
    vol_ok = True
    if volumes is not None and not volumes.empty:
        avg_vol = volumes.tail(config.lookback).mean()
        vol_ok = volumes.iloc[-1] >= avg_vol * config.volume_mult

    if last > recent_high and vol_ok:
        score = (last - recent_high) / (recent_high + 1e-9) * 100
        return Decision(market, last, max(score, config.min_score), Signal.BUY, f"돌파 매수 {recent_high:.1f}")
    if last < recent_low and vol_ok:
        score = -(recent_low - last) / (recent_low + 1e-9) * 100
        return Decision(market, last, min(score, -config.min_score), Signal.SELL, f"하락 돌파 {recent_low:.1f}")

    return Decision(market, last, 0.0, Signal.HOLD, "돌파 없음")
