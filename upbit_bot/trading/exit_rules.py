"""
공통 손절·익절·트레일링·시간 청산 로직.

TradingBot이 포지션 보유 여부와 독립적으로 활용할 수 있도록 유틸 함수로 분리했다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import atr_like, bollinger_bands


@dataclass
class ExitSignal:
    should_exit: bool
    reason: str
    stop_level: Optional[float] = None


def compute_stop_targets(prices: pd.Series, entry_price: float, *, atr_mult_stop: float = 1.6, atr_mult_target: float = 2.8) -> tuple[float, float]:
    atr = float(np.nan_to_num(atr_like(prices).iloc[-1]))
    stop_loss = max(0.0, entry_price - atr * atr_mult_stop)
    take_profit = entry_price + atr * atr_mult_target
    return stop_loss, take_profit


def evaluate_exit(
    prices: pd.Series,
    *,
    entry_price: float,
    entry_time: Optional[datetime],
    max_hold_hours: int = 72,
    trail_multiplier: float = 1.2,
) -> ExitSignal:
    if prices.empty:
        return ExitSignal(False, "가격 데이터 부족")

    last_price = float(prices.iloc[-1])
    stop_loss, take_profit = compute_stop_targets(prices, entry_price)

    if last_price <= stop_loss:
        return ExitSignal(True, f"ATR 기반 손절 {stop_loss:.2f} 하향 돌파", stop_loss)
    if last_price >= take_profit:
        return ExitSignal(True, f"ATR 기반 익절 {take_profit:.2f} 도달", take_profit)

    bb = bollinger_bands(prices)
    bb_upper = float(bb["upper"].iloc[-1])
    bb_middle = float(bb["middle"].iloc[-1])
    bb_lower = float(bb["lower"].iloc[-1])
    if last_price < bb_lower or last_price < bb_middle * 0.985:
        return ExitSignal(True, "볼린저 하단 이탈/중심선 재진입", bb_lower)

    rolling_high = float(prices.rolling(window=min(120, len(prices))).max().iloc[-1])
    trailing_stop = rolling_high - (atr_like(prices).iloc[-1] * trail_multiplier)
    if last_price <= trailing_stop:
        return ExitSignal(True, f"트레일링 스탑 {trailing_stop:.2f} 발동", trailing_stop)

    if entry_time:
        held_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
        if held_hours >= max_hold_hours:
            return ExitSignal(True, f"보유 시간 {held_hours:.1f}h가 한도 초과")

    return ExitSignal(False, "보유 유지")
