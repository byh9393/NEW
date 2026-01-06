"""
Regime-based strategy router.

- TREND: Supertrend + EMA filter (composite.evaluate)
- MEAN_REVERSION: Bollinger Z-score + RSI (mean_reversion.mean_reversion_signal)
- BREAKOUT: Donchian breakout + volume confirmation (breakout.breakout_signal)

The router selects a regime using simple, robust heuristics from OHLCV.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

import numpy as np
import pandas as pd

from upbit_bot.indicators import technical
from upbit_bot.strategy.composite import Decision, Signal, evaluate as evaluate_trend
from upbit_bot.strategy.mean_reversion import mean_reversion_signal, MeanRevConfig
from upbit_bot.strategy.breakout import breakout_signal, BreakoutConfig


class Regime(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


@dataclass
class RegimeThresholds:
    # Heuristics are expressed in normalized terms to be stable across price scales.
    # aggressive: switches more often to breakout/mean-reversion
    slope_trend: float
    bb_width_low: float
    bb_width_high: float
    breakout_proximity: float  # fraction from high/low for breakout candidate
    atr_norm_high: float


_THRESHOLDS: Dict[str, RegimeThresholds] = {
    "aggressive": RegimeThresholds(
        slope_trend=0.0015,
        bb_width_low=0.020,
        bb_width_high=0.060,
        breakout_proximity=0.015,
        atr_norm_high=0.020,
    ),
    "neutral": RegimeThresholds(
        slope_trend=0.0025,
        bb_width_low=0.018,
        bb_width_high=0.055,
        breakout_proximity=0.010,
        atr_norm_high=0.018,
    ),
    "conservative": RegimeThresholds(
        slope_trend=0.0035,
        bb_width_low=0.016,
        bb_width_high=0.050,
        breakout_proximity=0.007,
        atr_norm_high=0.016,
    ),
}


def _safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    needed = {"open", "high", "low", "close", "volume"}
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(needed))
    missing = needed.difference(frame.columns)
    if missing:
        # best-effort: construct missing columns from close where possible
        f = frame.copy()
        if "close" in f.columns:
            for col in missing:
                if col in ("open", "high", "low"):
                    f[col] = f["close"]
                elif col == "volume":
                    f[col] = 0.0
        return f
    return frame


def classify_regime(frame: pd.DataFrame, *, style: str = "aggressive") -> Regime:
    """Classify market regime from OHLCV frame."""
    f = _safe_frame(frame)
    if f.empty or len(f) < 60:
        return Regime.TREND

    style_key = (style or "aggressive").strip().lower()
    thr = _THRESHOLDS.get(style_key, _THRESHOLDS["aggressive"])

    close = f["close"].astype(float)
    # Trend strength proxy: EMA200 slope over recent window
    ema200 = technical.ema(close, 200)
    # Use last 25 bars for slope
    w = min(25, len(ema200.dropna()))
    if w < 10:
        slope = 0.0
    else:
        y = ema200.iloc[-w:].to_numpy(dtype=float)
        x = np.arange(w, dtype=float)
        # linear regression slope normalized by current price
        denom = (np.mean(y) + 1e-9)
        slope = float(np.polyfit(x, y, 1)[0] / denom)

    # Volatility proxies
    bb = technical.bollinger_bands(close, window=20, num_std=2.0)
    if bb.empty or bb["upper"].dropna().empty:
        bb_width = 0.0
    else:
        upper = float(bb["upper"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        mid = float(bb["middle"].iloc[-1])
        bb_width = float((upper - lower) / (abs(mid) + 1e-9))

    atr = technical.atr_like(close, period=14)
    atr_norm = float(atr.iloc[-1] / (abs(close.iloc[-1]) + 1e-9)) if not atr.empty else 0.0

    # Breakout proximity: close near recent high/low
    look = min(20, len(close))
    recent_high = float(f["high"].astype(float).iloc[-look:].max())
    recent_low = float(f["low"].astype(float).iloc[-look:].min())
    last = float(close.iloc[-1])
    dist_to_high = abs(recent_high - last) / (abs(recent_high) + 1e-9)
    dist_to_low = abs(last - recent_low) / (abs(recent_low) + 1e-9)

    # Aggressive: prioritize breakout when high volatility and near extremes
    near_extreme = (dist_to_high <= thr.breakout_proximity) or (dist_to_low <= thr.breakout_proximity)
    if near_extreme and (bb_width >= thr.bb_width_high or atr_norm >= thr.atr_norm_high):
        return Regime.BREAKOUT

    # Mean reversion: low volatility & weak trend
    if bb_width <= thr.bb_width_low and abs(slope) <= thr.slope_trend:
        return Regime.MEAN_REVERSION

    return Regime.TREND


def evaluate_regime_switch(
    market: str,
    frame: pd.DataFrame,
    *,
    style: str = "aggressive",
    enable_trend: bool = True,
    enable_mean_reversion: bool = True,
    enable_breakout: bool = True,
) -> Decision:
    """Route to a strategy based on regime classification."""
    f = _safe_frame(frame)
    close = f["close"].astype(float) if not f.empty and "close" in f.columns else pd.Series(dtype=float)
    if close.empty:
        return Decision(market, 0.0, 0.0, Signal.HOLD, "[router] 데이터 없음", 0.0, False, "router")

    regime = classify_regime(f, style=style)
    reason_prefix = f"[{regime.value.upper()}] "

    # Strategy configs: tuned to be responsive (aggressive defaults)
    if regime == Regime.BREAKOUT and enable_breakout:
        decision = breakout_signal(
            market,
            close,
            volumes=f.get("volume"),
            config=BreakoutConfig(lookback=20, volume_mult=1.15, min_score=40),
        )
        decision.reason = reason_prefix + decision.reason
        return decision

    if regime == Regime.MEAN_REVERSION and enable_mean_reversion:
        decision = mean_reversion_signal(
            market,
            close,
            config=MeanRevConfig(z_entry=0.85, rsi_low=38.0, rsi_high=62.0, min_score=40),
        )
        decision.reason = reason_prefix + decision.reason
        return decision

    if enable_trend:
        # Slightly more sensitive thresholds for aggressive style
        buy_th = 35 if (style or "").lower() == "aggressive" else 40
        sell_th = -35 if (style or "").lower() == "aggressive" else -40
        decision = evaluate_trend(market, close, buy_threshold=buy_th, sell_threshold=sell_th, frame=f)
        decision.reason = reason_prefix + decision.reason
        decision.strategy = "supertrend"
        return decision

    return Decision(market, float(close.iloc[-1]), 0.0, Signal.HOLD, reason_prefix + "전략 비활성", 0.0, False, "router")
