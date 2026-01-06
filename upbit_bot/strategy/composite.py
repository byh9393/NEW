"""
Supertrend 3단계(Weak/Medium/Strong)와 200EMA만으로 매수·매도 신호를 생성한다.

- Weak/Medium/Strong Supertrend 방향을 조합해 추세 상태를 구분
- 200EMA 상·하단 위치로 장기 추세 필터 적용
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import ema, supertrend

logger = logging.getLogger(__name__)


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Decision:
    market: str
    price: float
    score: float
    signal: Signal
    reason: str
    quality: float = 0.0
    suppress_log: bool = False
    strategy: str = "supertrend"


_SUPERTREND_CONFIGS: Tuple[Tuple[str, int, float], ...] = (
    ("weak", 7, 2.0),
    ("medium", 10, 3.0),
    ("strong", 14, 4.0),
)


def _supertrend_states(prices: pd.Series) -> Dict[str, Tuple[int, float]]:
    states: Dict[str, Tuple[int, float]] = {}
    for name, period, mult in _SUPERTREND_CONFIGS:
        st = supertrend(prices, period=period, multiplier=mult)
        direction = int(st["direction"].iloc[-1]) if not st["direction"].empty else 0
        level = float(st["supertrend"].iloc[-1]) if not st["supertrend"].empty else float("nan")
        states[name] = (direction, level)
    return states


def _describe_state(states: Dict[str, Tuple[int, float]]) -> str:
    weak_dir = states.get("weak", (0, 0.0))[0]
    medium_dir = states.get("medium", (0, 0.0))[0]
    strong_dir = states.get("strong", (0, 0.0))[0]

    if weak_dir == 1 and medium_dir == 1 and strong_dir == 1:
        return "상태 3: 강한 추세 (분할 익절/홀딩 권장)"
    if weak_dir == 1 and medium_dir == 1:
        return "상태 2: 추세 형성, 눌림목 매수 구간"
    if weak_dir == 1:
        return "상태 1: 약한 상승, 관찰"
    if weak_dir == -1 and medium_dir == -1 and strong_dir == -1:
        return "강한 하락 추세"
    if weak_dir == -1:
        return "상태 4: 추세 피로, 추가 매수 중단"
    return "중립"


def evaluate(market: str, prices: pd.Series, *, buy_threshold: float = 40, sell_threshold: float = -40, frame: Optional[pd.DataFrame] = None) -> Decision:
    """Supertrend·200EMA 기반 매수/매도 판단."""

    if prices.empty or prices.size < 20:
        return Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 부족", 0.0)

    latest_price = float(prices.iloc[-1])
    ema_long = ema(prices, 200).iloc[-1] if prices.size >= 50 else float("nan")
    st_states = _supertrend_states(prices)

    weak_dir = st_states.get("weak", (0, 0.0))[0]
    medium_dir = st_states.get("medium", (0, 0.0))[0]
    strong_dir = st_states.get("strong", (0, 0.0))[0]

    above_ema = latest_price > ema_long if not np.isnan(ema_long) else True
    state_desc = _describe_state(st_states)

    signal = Signal.HOLD
    score = 0.0
    reason = state_desc

    if above_ema:
        if weak_dir == 1 and medium_dir == 1 and strong_dir != 1:
            signal = Signal.BUY
            score = max(buy_threshold, 70.0)
            reason = "상태 2 진입: Weak+Medium 상승, 200EMA 상단"
        elif weak_dir == 1 and medium_dir == 1 and strong_dir == 1:
            signal = Signal.HOLD
            score = buy_threshold / 2
            reason = "상태 3: 강한 추세, 신규 진입 제한"
        elif weak_dir == -1:
            signal = Signal.SELL
            score = min(sell_threshold, -60.0)
            reason = "상태 4: Weak 하락 전환, 포지션 축소"
    else:
        if weak_dir == -1 and medium_dir == -1:
            signal = Signal.SELL
            score = min(sell_threshold, -70.0)
            reason = "200EMA 하단 하락 추세 지속"
        else:
            signal = Signal.HOLD
            reason = "장기 하락 구간 필터"

    decision = Decision(
        market=market,
        price=latest_price,
        score=score,
        signal=signal,
        reason=reason,
        quality=(latest_price - ema_long) / ema_long if not np.isnan(ema_long) else 0.0,
        suppress_log=False,
    )
    if not decision.suppress_log:
        logger.debug("%s 신호: %s", market, decision)
    return decision
