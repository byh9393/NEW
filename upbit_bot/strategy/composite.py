"""
여러 기술적 지표 점수를 합쳐 트레이딩 신호를 생성한다.

EMA 트렌드, RSI/Stochastic 모멘텀, 볼린저·켈트너 변동성/스퀴즈, ROC 속도,
Z-Score 평균회귀, Choppiness 시장 품질을 계층적으로 결합해 -100~100 점수를
계산한다. 점수가 양수일수록 매수 우위, 음수일수록 매도 우위를 의미한다.
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
    keltner_channel,
    percent_b,
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
    mid = ema(prices, 60).iloc[-1]
    slow = ema(prices, 200).iloc[-1] if prices.size >= 200 else ema(prices, 120).iloc[-1]
    fast_prev = ema(prices.shift(1).dropna(), 20).iloc[-1]
    mid_prev = ema(prices.shift(1).dropna(), 60).iloc[-1]

    bias_fast_mid = np.tanh((fast - mid) / (prices.std() + 1e-6)) * 35
    bias_mid_slow = np.tanh((mid - slow) / (prices.std() + 1e-6)) * 25
    slope_fast = np.clip((fast - fast_prev) / (fast_prev + 1e-9) * 1200, -18, 18)
    slope_mid = np.clip((mid - mid_prev) / (mid_prev + 1e-9) * 800, -12, 12)
    alignment = 12 if fast > mid > slow else -8 if fast < mid < slow else 0

    score = float(np.clip(bias_fast_mid + bias_mid_slow + slope_fast + slope_mid + alignment, -100, 100))
    direction = "상승" if score > 0 else "하락"
    return score, f"EMA 정렬 {direction} ({score:.1f})"


def _momentum_component(prices: pd.Series) -> Tuple[float, str]:
    rsi_val = rsi(prices).iloc[-1]
    rsi_slope = (rsi(prices).iloc[-1] - rsi(prices).iloc[-5]) / 5 if prices.size >= 5 else 0.0
    stoch = stochastic_oscillator(prices)
    stoch_k = stoch["%K"].iloc[-1]
    stoch_d = stoch["%D"].iloc[-1]
    roc_fast = rate_of_change(prices, period=6).iloc[-1]
    roc_mid = rate_of_change(prices, period=24).iloc[-1]
    accel = np.clip((roc_fast - roc_mid) * 0.6, -20, 20)

    rsi_score = np.clip((rsi_val - 50) * 1.4 + rsi_slope * 3, -40, 40)
    stoch_score = np.clip((stoch_k - stoch_d) / 100 * 28 + (stoch_k - 50) * 0.45, -30, 30)
    roc_score = np.clip(roc_fast * 0.6 + roc_mid * 0.25 + accel, -35, 35)
    score = float(np.clip(rsi_score + stoch_score + roc_score, -100, 100))
    heat = "강세" if score > 0 else "약세"
    return score, f"모멘텀 {heat} (RSI {rsi_val:.1f}/기울기 {rsi_slope:.2f}, ROC {roc_fast:.1f}/{roc_mid:.1f})"


def _volatility_component(prices: pd.Series) -> Tuple[float, str]:
    bb = bollinger_bands(prices)
    upper, middle, lower = bb.iloc[-1]["upper"], bb.iloc[-1]["middle"], bb.iloc[-1]["lower"]
    bandwidth = (upper - lower) / (middle + 1e-9)
    last = prices.iloc[-1]
    pos = (last - middle) / (upper - lower + 1e-6)
    kc = keltner_channel(prices)
    kc_upper, kc_lower = kc.iloc[-1]["upper"], kc.iloc[-1]["lower"]
    squeeze_on = (upper - lower) < (kc_upper - kc_lower) * 0.9

    breakout = np.clip(pos * 70, -70, 70)
    regime = np.clip((bandwidth - 0.025) * 850, -22, 22)
    squeeze_penalty = -18 if squeeze_on else 0
    score = float(np.clip(breakout + regime + squeeze_penalty, -100, 100))
    squeeze_text = "스퀴즈" if squeeze_on else "정상 변동성"
    return score, f"볼린저 {pos:.2f}, 밴드폭 {bandwidth:.3f}, {squeeze_text}"


def _mean_reversion_component(prices: pd.Series) -> Tuple[float, str]:
    z = zscore(prices).iloc[-1]
    pb = percent_b(prices).iloc[-1]
    z_score = float(np.clip(-z * 16, -28, 28))
    band_extreme = np.clip((0.5 - pb) * 55, -25, 25)
    score = float(np.clip(z_score + band_extreme, -60, 60))
    return score, f"Z-Score {z:.2f}, %B {pb:.2f} 평균회귀 {score:+.1f}"


def _quality_filter(prices: pd.Series) -> Tuple[float, str]:
    window = 60 if prices.size >= 60 else max(30, prices.size)
    rolling_max = prices.rolling(window=window).max()
    rolling_min = prices.rolling(window=window).min()
    tr = rolling_max - rolling_min
    sum_abs = prices.diff().abs().rolling(window=window).sum()
    choppiness = sum_abs / (tr + 1e-9)
    latest_chop = choppiness.iloc[-1]

    # 1.5 근처면 박스권, 1.2 이하면 추세 우위, 2 이상이면 과도한 노이즈
    quality = np.clip((1.5 - latest_chop) * 45, -35, 35)
    regime = "추세" if quality > 0 else "잡음/박스"
    return float(quality), f"시장 품질 {regime} (Chop {latest_chop:.2f})"


def _aggregate_factors(prices: pd.Series) -> Tuple[float, List[str], Dict[str, float]]:
    components = {
        "trend": _trend_component(prices),
        "momentum": _momentum_component(prices),
        "volatility": _volatility_component(prices),
        "mean_reversion": _mean_reversion_component(prices),
        "quality": _quality_filter(prices),
    }

    weighted_score = (
        components["trend"][0] * 0.32
        + components["momentum"][0] * 0.32
        + components["volatility"][0] * 0.16
        + components["mean_reversion"][0] * 0.1
        + components["quality"][0] * 0.1
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

    # 장기 하락 추세 및 시장 품질에 따른 필터 강화
    trend_score = raw_scores.get("trend", 0.0)
    quality_score = raw_scores.get("quality", 0.0)
    effective_buy = buy_threshold + (6 if trend_score < 0 else 0) + (4 if quality_score < -10 else 0)
    effective_sell = sell_threshold - (5 if trend_score < -25 else 0)

    if score >= effective_buy:
        signal = Signal.BUY
        reason = f"복합점수 {score:.1f} ≥ 매수 임계 {effective_buy} | " + "; ".join(reasons[:2])
    elif score <= effective_sell:
        signal = Signal.SELL
        reason = f"복합점수 {score:.1f} ≤ 매도 임계 {effective_sell} | " + "; ".join(reasons[:2])
    else:
        signal = Signal.HOLD
        reason = "중립 영역 | " + "; ".join(reasons[:2])

    decision = Decision(market=market, price=latest_price, score=score, signal=signal, reason=reason)
    logger.debug("%s 신호: %s", market, decision)
    return decision
