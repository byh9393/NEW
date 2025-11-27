"""
여러 기술적 지표 점수를 합쳐 트레이딩 신호를 생성한다.

EMA 트렌드, RSI/Stochastic 모멘텀, 볼린저 밴드 변동성/위치, ROC 속도, Z-Score
평균회귀 신호를 계층적으로 결합해 -100~100 점수를 계산한다. 점수가 양수일수록
매수 우위, 음수일수록 매도 우위를 의미한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from upbit_bot.indicators.technical import (
    bollinger_bands,
    ema,
    rate_of_change,
    rsi,
    stochastic_oscillator,
    zscore,
)

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


def _trend_component(prices: pd.Series) -> Tuple[float, str]:
    fast = ema(prices, 20).iloc[-1]
    slow = ema(prices, 60).iloc[-1]
    long = ema(prices, 200).iloc[-1] if prices.size >= 200 else ema(prices, 120).iloc[-1]
    recent_slope = (fast - ema(prices.shift(1).dropna(), 20).iloc[-1]) / (fast + 1e-9)

    bias_fast_slow = np.tanh((fast - slow) / (prices.std() + 1e-6)) * 40
    bias_slow_long = np.tanh((slow - long) / (prices.std() + 1e-6)) * 25
    slope_score = np.clip(recent_slope * 800, -15, 15)
    score = float(np.clip(bias_fast_slow + bias_slow_long + slope_score, -100, 100))
    direction = "상승" if score > 0 else "하락"
    return score, f"EMA 트렌드 {direction} ({score:.1f})"


def _momentum_component(prices: pd.Series) -> Tuple[float, str]:
    rsi_val = rsi(prices).iloc[-1]
    stoch = stochastic_oscillator(prices)
    stoch_k = stoch["%K"].iloc[-1]
    stoch_d = stoch["%D"].iloc[-1]

    rsi_score = np.clip((rsi_val - 50) * 1.2, -35, 35)
    stoch_score = np.clip((stoch_k - stoch_d) / 100 * 30 + (stoch_k - 50) * 0.4, -30, 30)
    roc_score = np.clip(rate_of_change(prices, period=24).iloc[-1], -20, 20)
    score = float(np.clip(rsi_score + stoch_score + roc_score, -100, 100))
    heat = "강세" if score > 0 else "약세"
    return score, f"모멘텀 {heat} (RSI {rsi_val:.1f}, Stoch {stoch_k:.1f}/{stoch_d:.1f})"


def _volatility_component(prices: pd.Series) -> Tuple[float, str]:
    bb = bollinger_bands(prices)
    upper, middle, lower = bb.iloc[-1]["upper"], bb.iloc[-1]["middle"], bb.iloc[-1]["lower"]
    bandwidth = (upper - lower) / (middle + 1e-9)
    last = prices.iloc[-1]
    pos = (last - middle) / (upper - lower + 1e-6)

    breakout = np.clip(pos * 70, -70, 70)
    regime = np.clip((bandwidth - 0.02) * 800, -20, 20)
    score = float(np.clip(breakout + regime, -100, 100))
    return score, f"볼린저 위치 {pos:.2f}, 밴드폭 {bandwidth:.3f}"


def _mean_reversion_component(prices: pd.Series) -> Tuple[float, str]:
    z = zscore(prices).iloc[-1]
    score = float(np.clip(-z * 18, -30, 30))  # z>0은 고평가 → 매도 압력
    return score, f"Z-Score {z:.2f} 기반 평균회귀 {score:+.1f}"


def _aggregate_factors(prices: pd.Series) -> Tuple[float, List[str], Dict[str, float]]:
    components = {
        "trend": _trend_component(prices),
        "momentum": _momentum_component(prices),
        "volatility": _volatility_component(prices),
        "mean_reversion": _mean_reversion_component(prices),
    }

    weighted_score = (
        components["trend"][0] * 0.35
        + components["momentum"][0] * 0.35
        + components["volatility"][0] * 0.2
        + components["mean_reversion"][0] * 0.1
    )
    score = float(np.clip(weighted_score, -100, 100))
    reasons = [comp[1] for comp in components.values()]
    raw_scores = {name: comp[0] for name, comp in components.items()}
    return score, reasons, raw_scores


def evaluate(market: str, prices: pd.Series, *, buy_threshold: float = 25, sell_threshold: float = -25) -> Decision:
    """
    시장의 최근 가격 히스토리를 바탕으로 매수/매도 신호 평가.

    - 트렌드/모멘텀/변동성/평균회귀 네 가지 축을 가중 합산
    - 시스템 트레이더가 선호하는 필터링: 장기 하락 추세에서는 매수 한도를 낮춤
    """
    if prices.empty or prices.size < 60:
        return Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 부족")

    score, reasons, raw_scores = _aggregate_factors(prices)
    latest_price = float(prices.iloc[-1])

    # 장기 하락 추세에서는 매수 필터 강화
    trend_score = raw_scores.get("trend", 0.0)
    effective_buy = buy_threshold + (5 if trend_score < 0 else 0)

    if score >= effective_buy:
        signal = Signal.BUY
        reason = f"복합점수 {score:.1f} ≥ 매수 임계 {effective_buy} | " + "; ".join(reasons[:2])
    elif score <= sell_threshold:
        signal = Signal.SELL
        reason = f"복합점수 {score:.1f} ≤ 매도 임계 {sell_threshold} | " + "; ".join(reasons[:2])
    else:
        signal = Signal.HOLD
        reason = "중립 영역 | " + "; ".join(reasons[:2])

    decision = Decision(market=market, price=latest_price, score=score, signal=signal, reason=reason)
    logger.debug("%s 신호: %s", market, decision)
    return decision
