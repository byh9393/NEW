"""Upbit 시장 목록을 가져오고 필터링하는 유틸리티.

"마켓 조회가 오래 걸리는" 케이스는 대부분 API 응답 자체가 느려서가 아니라,
클라이언트 환경의 네트워크 설정(프록시 자동탐지, IPv6 우선 연결) 때문에
초기 연결이 지연되는 경우가 많다.

이 파일은 2가지를 동시에 달성한다.

1) 네트워크가 정상일 때는 빠르게 /market/all → /ticker(필요시) 순으로 완료
2) 네트워크가 비정상/지연일 때도 캐시를 이용해 즉시 시작할 수 있도록 보장
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from upbit_bot.data.upbit_adapter import UpbitAdapter
from upbit_bot.indicators.technical import (
    atr_like,
    ema,
    macd,
    percent_b,
    rate_of_change,
    rsi,
)

logger = logging.getLogger(__name__)


def _default_cache_path() -> Path:
    return Path("./.cache/markets/krw_markets.json")


def _load_cache(path: Path, *, max_age_sec: int) -> Tuple[List[str], Dict[str, float]] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = float(data.get("ts", 0.0))
        if ts <= 0:
            return None
        if (time.time() - ts) > max_age_sec:
            return None
        markets = [str(x) for x in data.get("markets", []) if str(x)]
        volumes = {str(k): float(v) for k, v in (data.get("volumes") or {}).items()}
        if not markets:
            return None
        return markets, volumes
    except Exception:
        logger.debug("마켓 캐시 로드 실패", exc_info=True)
        return None


def _save_cache(path: Path, *, markets: List[str], volumes: Dict[str, float]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.time(), "markets": markets, "volumes": volumes}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.debug("마켓 캐시 저장 실패", exc_info=True)


def _fetch_24h_volumes(
    markets: Sequence[str], adapter: UpbitAdapter, *, deadline: float | None = None
) -> Dict[str, float]:
    """요청한 마켓들의 24시간 거래대금을 반환한다.

    마켓 수가 많을 때 여러 번의 REST 호출이 필요하므로, ``deadline`` 을 넘기면
    이후 배치는 건너뛰어 UI의 전체 타임아웃을 넘지 않도록 방어한다.
    """

    volumes: Dict[str, float] = {}
    for idx in range(0, len(markets), 100):
        if deadline and time.monotonic() >= deadline:
            logger.warning("24h 거래대금 조회가 지연되어 일부 배치를 건너뜁니다.")
            break

        batch = markets[idx : idx + 100]
        try:
            tickers = adapter.ticker(batch, deadline=deadline, rate_limit_wait=False)
        except Exception:
            # 특정 배치가 실패하더라도 나머지 배치를 계속 진행해 상위 N개 추출이
            # 완전히 무산되지 않도록 방어한다.
            logger.exception("%d번째 배치 거래대금 조회 실패. 건너뜀", idx // 100)
            continue
        for ticker in tickers:
            market = ticker.get("market")
            volume = float(ticker.get("acc_trade_price_24h", 0.0))
            if market:
                volumes[market] = volume
    return volumes


@dataclass
class ScalpFilterConfig:
    """단타 매매 후보를 추리기 위한 필터 구성값."""

    unit: int = 5
    candle_count: int = 120
    min_abs_score: float = 30.0
    top_n: int = 12


def _candles_to_series(candles: Sequence[dict]) -> pd.Series:
    prices = [float(candle.get("trade_price", 0.0) or 0.0) for candle in reversed(candles)]
    return pd.Series(prices, dtype=float)


def _scalp_score(prices: pd.Series) -> float:
    """단타 진입/청산 탐지용 지표 스코어를 계산한다."""
    if prices.size < 30:
        return 0.0

    last_price = float(prices.iloc[-1])
    ema_fast = float(ema(prices, 9).iloc[-1])
    ema_slow = float(ema(prices, 21).iloc[-1])
    rsi_val = float(rsi(prices, 14).iloc[-1])
    macd_hist = float(macd(prices)["hist"].iloc[-1])
    percent_b_val = float(percent_b(prices, 20).iloc[-1])
    roc_val = float(rate_of_change(prices, 5).iloc[-1])
    atr_val = float(atr_like(prices, 14).iloc[-1])

    score = 0.0
    score += 15.0 if ema_fast > ema_slow else -15.0
    score += float(np.tanh(macd_hist / (prices.std() + 1e-6)) * 20)

    if rsi_val < 35:
        score += 20.0
    elif rsi_val > 65:
        score -= 20.0

    if percent_b_val < 0.2:
        score += 15.0
    elif percent_b_val > 0.8:
        score -= 15.0

    if roc_val > 0.6:
        score += 10.0
    elif roc_val < -0.6:
        score -= 10.0

    if last_price > 0:
        volatility_ratio = atr_val / last_price
        if volatility_ratio < 0.001:
            score *= 0.6
        elif volatility_ratio > 0.02:
            score *= 0.8

    return float(np.clip(score, -100, 100))


def _filter_by_scalp_signals(
    markets: Sequence[str],
    adapter: UpbitAdapter,
    *,
    config: ScalpFilterConfig,
    deadline: float | None = None,
) -> List[str]:
    scored: List[Tuple[str, float]] = []
    for market in markets:
        if deadline and time.monotonic() >= deadline:
            logger.warning("단타 지표 필터 시간이 초과되어 일부 종목만 적용합니다.")
            break
        try:
            candles = adapter.candles(
                market=market,
                kind="minutes",
                unit=config.unit,
                count=config.candle_count,
                deadline=deadline,
            )
        except Exception:
            logger.exception("단타 캔들 조회 실패: %s", market)
            continue
        if not candles:
            continue
        prices = _candles_to_series(candles)
        score = _scalp_score(prices)
        if abs(score) >= config.min_abs_score:
            scored.append((market, score))

    if not scored:
        return []

    scored_sorted = sorted(scored, key=lambda item: abs(item[1]), reverse=True)
    if config.top_n > 0:
        scored_sorted = scored_sorted[: config.top_n]
    return [market for market, _ in scored_sorted]


def fetch_markets(
    *,
    is_fiat: bool = True,
    fiat_symbol: str = "KRW",
    top_by_volume: int | None = None,
    scalp_filter: ScalpFilterConfig | None = None,
    time_budget: float | None = None,
    adapter: UpbitAdapter | None = None,
    cache_path: str | Path | None = None,
    cache_max_age_sec: int = 60 * 60 * 24,
) -> List[str]:
    """업비트의 거래가능 시장 목록을 조회한다."""

    cache_file = Path(cache_path) if cache_path is not None else _default_cache_path()
    cached = _load_cache(cache_file, max_age_sec=cache_max_age_sec)
    if cached is not None:
        cached_markets, cached_volumes = cached
        if top_by_volume:
            cached_sorted = sorted(cached_markets, key=lambda m: cached_volumes.get(m, 0.0), reverse=True)
            return cached_sorted[:top_by_volume]
        return cached_markets

    client = adapter or UpbitAdapter()
    deadline = (
        time.monotonic() + time_budget
        if time_budget is not None and time_budget > 0
        else None
    )

    t0 = time.perf_counter()
    markets: Sequence[dict] = client.list_markets(
        is_details=False,
        deadline=deadline,
        rate_limit_wait=False,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if elapsed > 2.0:
        logger.warning(
            "market/all 응답이 느립니다(%.2fs). 프록시 자동탐지/IPv6 우선 연결 여부를 확인하세요.",
            elapsed,
        )

    if not is_fiat:
        base_markets = [market["market"] for market in markets]
    else:
        base_markets = [market["market"] for market in markets if market["market"].startswith(fiat_symbol)]

    logger.info("가져온 시장 수: %d (market/all %.2fs)", len(base_markets), t1 - t0)

    volumes: Dict[str, float] = {}
    if top_by_volume:
        vol_deadline = deadline or (time.monotonic() + 8)
        t2 = time.perf_counter()
        try:
            volumes = _fetch_24h_volumes(base_markets, client, deadline=vol_deadline)
        except Exception:
            logger.exception("거래대금 순위 조회 중 오류. 상위 %d개로만 제한합니다.", top_by_volume)
            volumes = {}
        t3 = time.perf_counter()

        selected = sorted(base_markets, key=lambda m: volumes.get(m, 0.0), reverse=True)[:top_by_volume]
        logger.info("24시간 거래대금 상위 %d개 시장으로 제한 (ticker %.2fs)", len(selected), t3 - t2)
    else:
        selected = base_markets

    # 네트워크로 성공적으로 받아온 경우 캐시에 저장
    _save_cache(cache_file, markets=list(base_markets), volumes=volumes)

    if scalp_filter is None:
        return selected

    scalp_markets = _filter_by_scalp_signals(
        selected,
        client,
        config=scalp_filter,
        deadline=deadline,
    )
    if scalp_markets:
        logger.info("단타 지표 필터 적용: %d개 종목 선정", len(scalp_markets))
        return scalp_markets

    logger.warning("단타 지표 필터에서 종목을 찾지 못해 원본 목록을 사용합니다.")
    return selected
