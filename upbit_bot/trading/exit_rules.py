"""
Supertrend와 200EMA만 사용하는 포지션 청산 규칙.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import ema, supertrend


@dataclass
class ExitSignal:
    should_exit: bool
    reason: str
    stop_level: Optional[float] = None


def compute_stop_targets(prices: pd.Series, entry_price: float, *, risk_reward: float = 1.8) -> tuple[float, float]:
    st = supertrend(prices, period=7, multiplier=2.0)
    st_level = float(st["supertrend"].iloc[-1]) if not st.empty else entry_price * 0.99
    stop_loss = max(0.0, min(entry_price, st_level))
    take_profit = entry_price + (entry_price - stop_loss) * risk_reward
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

    ema_long = ema(prices, 200).iloc[-1] if prices.size >= 50 else float("nan")
    st_dir = int(supertrend(prices, period=10, multiplier=3.0)["direction"].iloc[-1]) if prices.size >= 20 else 0

    if last_price <= stop_loss:
        return ExitSignal(True, f"Supertrend 기반 손절 {stop_loss:.2f} 하향 돌파", stop_loss)
    if last_price >= take_profit:
        return ExitSignal(True, f"Supertrend 기반 익절 {take_profit:.2f} 도달", take_profit)

    if st_dir == -1:
        return ExitSignal(True, "Supertrend 하락 전환", stop_loss)

    if not np.isnan(ema_long) and last_price < ema_long:
        return ExitSignal(True, "200EMA 하향 이탈", ema_long)

    if entry_time:
        held_hours = (datetime.utcnow() - entry_time).total_seconds() / 3600
        if held_hours >= max_hold_hours:
            return ExitSignal(True, f"보유 시간 {held_hours:.1f}h가 한도 초과")

    return ExitSignal(False, "보유 유지")
