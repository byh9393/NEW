"""
OpenAI를 활용해 기술적 지표와 가격 흐름을 요약하고 최종 매수/매도 신호를 산출한다.
다중 기술적 지표 전략을 LLM에 설명해 전문가형 판단을 보강한다.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from upbit_bot.indicators.technical import (
    bollinger_bands,
    ema,
    keltner_channel,
    macd,
    percent_b,
    rate_of_change,
    rsi,
    stochastic_oscillator,
    zscore,
)
from upbit_bot.strategy.composite import Decision, Signal, evaluate as evaluate_composite

logger = logging.getLogger(__name__)


@dataclass
class AIDecision:
    decision: Decision
    raw_response: str


def _format_metrics(prices: pd.Series) -> dict:
    macd_df = macd(prices)
    bb = bollinger_bands(prices)
    kc = keltner_channel(prices)
    rsi_series = rsi(prices)
    stoch = stochastic_oscillator(prices)
    fast = ema(prices, 20)
    slow = ema(prices, 60)
    long = ema(prices, 200) if prices.size >= 200 else ema(prices, 120)
    returns_1m = prices.pct_change(60).iloc[-1] * 100 if prices.size >= 61 else 0.0
    returns_5m = prices.pct_change(300).iloc[-1] * 100 if prices.size >= 301 else 0.0
    vol_5m = prices.pct_change().tail(300).std() * np.sqrt(60)
    roc_24 = rate_of_change(prices, period=24).iloc[-1]
    z_val = zscore(prices).iloc[-1]
    bandwidth = (bb["upper"].iloc[-1] - bb["lower"].iloc[-1]) / (bb["middle"].iloc[-1] + 1e-9)
    pb = percent_b(prices).iloc[-1]
    kc_width = (kc["upper"].iloc[-1] - kc["lower"].iloc[-1]) / (kc["middle"].iloc[-1] + 1e-9)
    squeeze_on = (bb["upper"].iloc[-1] - bb["lower"].iloc[-1]) < (kc["upper"].iloc[-1] - kc["lower"].iloc[-1]) * 0.9
    window = 60 if prices.size >= 60 else max(30, prices.size)
    roll_max = prices.rolling(window=window).max()
    roll_min = prices.rolling(window=window).min()
    tr = roll_max - roll_min
    sum_abs = prices.diff().abs().rolling(window=window).sum()
    choppiness = float((sum_abs / (tr + 1e-9)).iloc[-1]) if prices.size else 0.0

    return {
        "latest_price": float(prices.iloc[-1]),
        "macd_hist": float(macd_df["hist"].iloc[-1]),
        "rsi": float(rsi_series.iloc[-1]),
        "stoch_k": float(stoch["%K"].iloc[-1]),
        "stoch_d": float(stoch["%D"].iloc[-1]),
        "bb_upper": float(bb["upper"].iloc[-1]),
        "bb_lower": float(bb["lower"].iloc[-1]),
        "bb_middle": float(bb["middle"].iloc[-1]),
        "bb_bandwidth": float(bandwidth),
        "bb_percent_b": float(pb),
        "ema_fast": float(fast.iloc[-1]),
        "ema_slow": float(slow.iloc[-1]),
        "ema_long": float(long.iloc[-1]),
        "returns_1m": float(returns_1m),
        "returns_5m": float(returns_5m),
        "roc_24": float(roc_24),
        "zscore": float(z_val),
        "volatility_5m": float(vol_5m),
        "kc_width": float(kc_width),
        "squeeze_on": bool(squeeze_on),
        "choppiness": float(choppiness),
    }


def _build_prompt(market: str, prices: pd.Series, base: Decision, metrics: dict) -> str:
    history_tail = ", ".join(f"{p:.2f}" for p in prices.tail(10))
    lines = [
        "너는 포지션 사이징과 진입 타이밍을 중시하는 한국인 시스템 트레이더다. 아래 데이터를 근거로 **다중 지표 합성** 관점에서 최종 매수/매도/대기 신호를 정리해라.",
        f"- 시장: {market}",
        f"- 최근 10틱 가격: {history_tail}",
        f"- 최근가: {metrics['latest_price']:.2f}",
        f"- 1분/5분 수익률(%): {metrics['returns_1m']:.3f} / {metrics['returns_5m']:.3f}",
        f"- MACD 히스토그램: {metrics['macd_hist']:.5f}",
        f"- EMA(20/60/200): {metrics['ema_fast']:.2f} / {metrics['ema_slow']:.2f} / {metrics['ema_long']:.2f}",
        f"- RSI: {metrics['rsi']:.2f}",
        f"- Stochastic %K/%D: {metrics['stoch_k']:.2f} / {metrics['stoch_d']:.2f}",
        f"- ROC(24틱 %): {metrics['roc_24']:.2f}",
        f"- 볼린저 밴드: 상단 {metrics['bb_upper']:.2f}, 중단 {metrics['bb_middle']:.2f}, 하단 {metrics['bb_lower']:.2f}, 밴드폭 {metrics['bb_bandwidth']:.4f}, %B {metrics['bb_percent_b']:.3f}",
        f"- Keltner 폭: {metrics['kc_width']:.4f}, Squeeze: {metrics['squeeze_on']}",
        f"- Z-Score: {metrics['zscore']:.3f}",
        f"- 5분 변동성(표준편차, %): {metrics['volatility_5m']:.3f}",
        f"- 시장 난이도(Choppiness): {metrics['choppiness']:.3f}",
        f"- 기본 모델 판단: {base.signal} (점수 {base.score:.1f}, 이유: {base.reason})",
        "",
        "지침:",
        "1) 추세 필터: EMA 20/60/200 정렬, MACD 히스토그램 방향, ROC 24/가속도를 함께 보고 순도 높은 추세만 통과시켜라.",
        "2) 모멘텀 확인: RSI·Stochastic의 레벨과 기울기, 단기/중기 ROC 간의 가속을 언급하라.",
        "3) 변동성·포지션 관리: 볼린저·Keltner 스퀴즈 여부, 밴드 상/하단 돌파 맥락을 명확히 진술하라.",
        "4) 평균회귀: Z-Score 절댓값이 2 이상 또는 %B 0.1/0.9 외곽이면 과열·과매도 경고.",
        "5) 시장 품질: Choppiness가 높으면 관망을 우선 검토하라.",
        "6) 신호는 BUY/SELL/HOLD 중 하나를 선택하고, confidence(0~100)를 근거와 함께 제시.",
        "",
        "출력은 JSON 한 줄만 작성한다. 키 이름은 반드시 signal(BUY/SELL/HOLD), confidence(0~100), reason(짧게) 세 개만 사용하라.",
        "예시: {\"signal\": \"BUY\", \"confidence\": 78, \"reason\": \"상승 추세+RSI50 상향+밴드 상단 돌파\"}",
    ]
    return "\n".join(lines)


def _parse_signal(raw: str, fallback: Decision) -> Decision:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    content = match.group(0) if match else raw
    try:
        parsed = json.loads(content)
        signal_text = str(parsed.get("signal", "")).upper()
        if signal_text not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("invalid signal")
        confidence = float(parsed.get("confidence", fallback.score))
        reason = str(parsed.get("reason", "")) or "LLM 판단"
        signal = Signal(signal_text)
        score = max(min(confidence, 100), -100)
        return Decision(
            market=fallback.market,
            price=fallback.price,
            score=score,
            signal=signal,
            reason=reason,
            strategy="ai",
            quality=fallback.quality,
        )
    except Exception:
        logger.warning("LLM 응답 파싱 실패, 기본 결정 사용. raw=%s", raw)
        return fallback


def evaluate_with_openai(
    market: str,
    prices: pd.Series,
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    base_decision: Optional[Decision] = None,
) -> AIDecision:
    """
    기본 기술적 점수 결정에 OpenAI 판단을 결합한다.

    OpenAI API 키가 없거나 통신 오류가 발생하면 기본 결정을 반환한다.
    """
    if prices.empty:
        empty_decision = base_decision or Decision(market, 0.0, 0.0, Signal.HOLD, "데이터 없음", 0.0, False, "ai")
        return AIDecision(decision=empty_decision, raw_response="")

    base = base_decision or evaluate_composite(market, prices)
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        logger.info("OPENAI_API_KEY가 설정되지 않아 기본 신호를 사용합니다.")
        return AIDecision(decision=base, raw_response="")

    client = OpenAI(api_key=key)
    metrics = _format_metrics(prices)
    prompt = _build_prompt(market, prices, base, metrics)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content if response.choices else ""
        if not raw_text:
            raise ValueError("빈 응답")
        decision = _parse_signal(raw_text, base)
        return AIDecision(decision=decision, raw_response=raw_text)
    except Exception:
        logger.exception("OpenAI 평가 실패, 기본 신호를 사용합니다.")
        return AIDecision(decision=base, raw_response="")
