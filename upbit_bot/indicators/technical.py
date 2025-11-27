"""
여러 기술적 지표 계산 함수.

트렌드(EMA), 모멘텀(RSI, Stochastic), 변동성(볼린저 밴드), 모멘텀 속도(ROC) 등
시스템 트레이더들이 다층 필터로 사용하는 지표를 중심으로 구성했다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic_oscillator(series: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """단일 종가 시퀀스를 고점/저점 근사로 사용해 Stochastic %K/%D를 계산한다."""
    lowest_low = series.rolling(window=k_period).min()
    highest_high = series.rolling(window=k_period).max()
    percent_k = (series - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
    percent_d = percent_k.rolling(window=d_period).mean()
    return pd.DataFrame({"%K": percent_k, "%D": percent_d})


def rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    """가격 변동 속도를 % 단위로 측정한다."""
    return series.pct_change(periods=period) * 100


def zscore(series: pd.Series, lookback: int = 50) -> pd.Series:
    rolling_mean = series.rolling(window=lookback).mean()
    rolling_std = series.rolling(window=lookback).std()
    return (series - rolling_mean) / (rolling_std + 1e-9)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram})


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    ma = sma(series, period)
    std = series.rolling(window=period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return pd.DataFrame({"upper": upper, "middle": ma, "lower": lower})


def composite_score(price_history: pd.Series) -> float:
    """
    여러 지표를 결합해 -100 ~ 100 사이의 점수로 정규화.
    양수는 매수 우위, 음수는 매도 우위를 의미.
    """
    if price_history.size < 50:
        return 0.0

    macd_df = macd(price_history)
    macd_signal = np.tanh(macd_df["hist"].iloc[-1] / (price_history.std() + 1e-6)) * 50

    rsi_val = rsi(price_history).iloc[-1]
    rsi_signal = (rsi_val - 50)  # -50 ~ 50 근사

    bb = bollinger_bands(price_history)
    last_price = price_history.iloc[-1]
    upper, lower = bb["upper"].iloc[-1], bb["lower"].iloc[-1]
    width = upper - lower if upper and lower else price_history.std() * 2
    bollinger_signal = 50 * (last_price - bb["middle"].iloc[-1]) / (width + 1e-6)

    return float(np.clip(macd_signal + rsi_signal + bollinger_signal, -100, 100))
