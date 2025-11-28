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
from time import time
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
    quality: float = 0.0
    suppress_log: bool = False


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


_risk_state: Dict[str, Dict[str, float]] = {}


def _cooldown_check(market: str, prices: pd.Series) -> Tuple[bool, str, float]:
    """최근 손실/드로다운 흐름을 기반으로 거래를 일시 중단한다."""

    state = _risk_state.setdefault(market, {"cooldown_until": 0.0})
    now = time()

    returns = prices.pct_change().dropna().tail(6)
    loss_streak = 0
    for r in reversed(returns.tolist()):
        if r < 0:
            loss_streak += 1
        else:
            break

    recent_peak = prices.rolling(window=min(90, prices.size)).max().iloc[-1]
    drawdown = (prices.iloc[-1] - recent_peak) / (recent_peak + 1e-9)

    cooldown_reason = ""
    if loss_streak >= 3:
        state["cooldown_until"] = max(state["cooldown_until"], now + 180)
        cooldown_reason = f"최근 {loss_streak}연속 손실"
    if drawdown <= -0.05:
        state["cooldown_until"] = max(state["cooldown_until"], now + 300)
        cooldown_reason = (cooldown_reason + " / " if cooldown_reason else "") + "단기 드로다운 -5% 이하"

    active = now < state.get("cooldown_until", 0)
    return active, cooldown_reason, drawdown


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
        return Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 부족", 0.0)

    score, reasons, raw_scores = _aggregate_factors(prices)
    latest_price = float(prices.iloc[-1])

    # 장기 하락 추세 및 시장 품질에 따른 필터 강화 + 거래비용 보정
    trend_score = raw_scores.get("trend", 0.0)
    quality_score = raw_scores.get("quality", 0.0)
    atr = float(np.nan_to_num(prices.diff().abs().rolling(window=14).mean().iloc[-1]))
    atr_ratio = atr / (latest_price + 1e-9)
    fee_rate = 0.0005
    slippage_est = min(0.002, atr_ratio * 1.6)
    cost_buffer = (fee_rate + slippage_est) * 140  # 점수 스케일로 변환
    volatility_buffer = np.clip(atr_ratio * 120, 0, 10)
    edge_score = score - cost_buffer * 0.6 - volatility_buffer * 0.8

    # 주요 팩터가 한 방향으로 강하게 모여 있는지 확인해 약한 잡음을 배제한다.
    positive_components = sum(
        raw_scores[name] > 12 for name in ("trend", "momentum", "volatility")
    ) + (1 if raw_scores.get("quality", 0.0) > 5 else 0)
    negative_components = sum(
        raw_scores[name] < -12 for name in ("trend", "momentum", "volatility")
    ) + (1 if raw_scores.get("quality", 0.0) < -5 else 0)
    mean_reversion_bias = raw_scores.get("mean_reversion", 0.0)
    strong_buy_alignment = positive_components >= 3 and mean_reversion_bias > -12
    strong_sell_alignment = negative_components >= 3 and mean_reversion_bias < 12

    effective_buy = (
        buy_threshold
        + (6 if trend_score < 0 else 0)
        + (4 if quality_score < -10 else 0)
        + cost_buffer
        + volatility_buffer
    )
    effective_sell = (
        sell_threshold
        - (5 if trend_score < -25 else 0)
        - (cost_buffer * 0.6)
        - (volatility_buffer * 0.5)
    )

    # 손절/익절/트레일링 스탑 탐지 (ATR·볼린저 기반)
    bb = bollinger_bands(prices)
    bb_upper, bb_middle, bb_lower = bb.iloc[-1]["upper"], bb.iloc[-1]["middle"], bb.iloc[-1]["lower"]
    trailing_peak = prices.rolling(window=min(60, prices.size)).max().iloc[-1]
    trailing_drawdown = (latest_price - trailing_peak) / (trailing_peak + 1e-9)

    stop_reason: Optional[str] = None
    if latest_price < bb_lower or latest_price < bb_middle - 1.5 * atr:
        stop_reason = "볼린저 하단/ATR 손절 발동"
    elif latest_price > bb_upper and score < effective_buy:
        stop_reason = "볼린저 상단 도달 후 익절/청산"
    elif trailing_drawdown <= -0.035:
        stop_reason = "트레일링 스탑: 최근 고점 대비 3.5% 하락"

    # 연속 손실/드로다운 기반 쿨다운 확인
    cooldown_active, cooldown_reason, drawdown = _cooldown_check(market, prices)
    suppress_log = False
    if stop_reason:
        signal = Signal.SELL
        reason = f"리스크 종료 신호: {stop_reason}"
    elif abs(edge_score) < 10:
        signal = Signal.HOLD
        reason = f"수수료/슬리피지 대비 기대 수익 부족 (엣지 {edge_score:.1f})"
        suppress_log = True
    elif cooldown_active:
        signal = Signal.HOLD
        reason = f"쿨다운: {cooldown_reason or f'DD {drawdown*100:.1f}%'}"
    elif score >= effective_buy and strong_buy_alignment and edge_score >= 12:
        signal = Signal.BUY
        reason = (
            f"강한 매수 합의: 복합점수 {score:.1f} ≥ {effective_buy:.1f}, "
            f"동행 팩터 {positive_components}개, 평균회귀 {mean_reversion_bias:.1f}"
        )
    elif score <= effective_sell and strong_sell_alignment and edge_score <= -12:
        signal = Signal.SELL
        reason = (
            f"강한 매도 합의: 복합점수 {score:.1f} ≤ {effective_sell:.1f}, "
            f"동행 팩터 {negative_components}개, 평균회귀 {mean_reversion_bias:.1f}"
        )
    else:
        signal = Signal.HOLD
        reason = "중립 또는 합의 부족 | " + "; ".join(reasons[:2])
        suppress_log = True

    decision = Decision(
        market=market,
        price=latest_price,
        score=score,
        signal=signal,
        reason=reason,
        quality=quality_score,
        suppress_log=suppress_log,
    )
    if not decision.suppress_log:
        logger.debug("%s 신호: %s", market, decision)
    return decision
