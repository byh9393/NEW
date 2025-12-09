"""
유니버스 동적 관리 모듈.

- 30일 평균 일 거래대금 기준 필터링
- 24시간 거래대금 상위 N개 시장 제한
- 호가 스프레드가 과도한 종목 제외
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ORDERBOOK_URL = "https://api.upbit.com/v1/orderbook"


@dataclass
class UniverseSnapshot:
    timestamp: datetime
    eligible: List[str]
    turnover_24h: Dict[str, float]
    turnover_30d_avg: Dict[str, float]
    spreads: Dict[str, float]


def _chunked(items: List[str], size: int = 30) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_spreads(markets: List[str]) -> Dict[str, float]:
    """주문호가 상단 스프레드를 가져와 종목별 상대 스프레드(%)를 계산한다."""

    spreads: Dict[str, float] = {}
    for batch in _chunked(markets, 30):
        try:
            resp = requests.get(ORDERBOOK_URL, params={"markets": ",".join(batch)}, timeout=10)
            resp.raise_for_status()
        except Exception:
            logger.exception("호가 스프레드 조회 실패")
            continue

        for ob in resp.json():
            code = ob.get("code")
            units = ob.get("orderbook_units") or []
            if not code or not units:
                continue
            top = units[0]
            bid = float(top.get("bid_price", 0.0) or 0.0)
            ask = float(top.get("ask_price", 0.0) or 0.0)
            mid = (bid + ask) / 2 if (bid and ask) else 0.0
            if bid <= 0 or ask <= 0 or mid <= 0:
                continue
            spread = (ask - bid) / mid
            spreads[code] = spread
    return spreads


class UniverseManager:
    def __init__(
        self,
        *,
        min_30d_avg_turnover: float = 1_000_000_000.0,
        max_spread_pct: float = 0.02,
        top_n: int = 10,
        refresh_interval: timedelta = timedelta(hours=1),
    ) -> None:
        self.min_30d_avg_turnover = min_30d_avg_turnover
        self.max_spread_pct = max_spread_pct
        self.top_n = top_n
        self.refresh_interval = refresh_interval
        self.last_snapshot: Optional[UniverseSnapshot] = None

    def _turnover_from_frame(self, frame: pd.DataFrame) -> tuple[float, float]:
        if frame.empty:
            return 0.0, 0.0
        turnover_series = frame["close"] * frame.get("volume", 0.0)
        avg_30d = float(turnover_series.tail(30).mean()) if len(turnover_series) else 0.0
        last_24h = float(turnover_series.iloc[-1])
        return avg_30d, last_24h

    def _filter_eligible(
        self,
        markets: List[str],
        frames: Dict[str, pd.DataFrame],
        spreads: Optional[Dict[str, float]],
    ) -> UniverseSnapshot:
        turnover_24h: Dict[str, float] = {}
        turnover_30d: Dict[str, float] = {}
        eligible: List[str] = []

        for market in markets:
            frame = frames.get(market, pd.DataFrame())
            avg_30d, last_24h = self._turnover_from_frame(frame)
            turnover_30d[market] = avg_30d
            turnover_24h[market] = last_24h
            if avg_30d < self.min_30d_avg_turnover:
                continue
            if spreads and market in spreads:
                if spreads[market] > self.max_spread_pct:
                    continue
            eligible.append(market)

        eligible_sorted = sorted(eligible, key=lambda m: turnover_24h.get(m, 0.0), reverse=True)
        if self.top_n > 0:
            eligible_sorted = eligible_sorted[: self.top_n]

        return UniverseSnapshot(
            timestamp=datetime.utcnow(),
            eligible=eligible_sorted,
            turnover_24h=turnover_24h,
            turnover_30d_avg=turnover_30d,
            spreads=spreads or {},
        )

    def refresh(
        self,
        *,
        markets: List[str],
        frame_lookup: Dict[str, pd.DataFrame],
        spreads: Optional[Dict[str, float]] = None,
        fetch_spread_if_missing: bool = False,
    ) -> UniverseSnapshot:
        if spreads is None and fetch_spread_if_missing:
            spreads = fetch_spreads(markets)

        snapshot = self._filter_eligible(markets, frame_lookup, spreads)
        self.last_snapshot = snapshot
        logger.info(
            "유니버스 갱신: %d/%d 종목 채택 (30일 평균 거래대금 ≥ %.0f, 스프레드 ≤ %.2f%%)",
            len(snapshot.eligible),
            len(markets),
            self.min_30d_avg_turnover,
            self.max_spread_pct * 100,
        )
        return snapshot

    def should_refresh(self) -> bool:
        if self.last_snapshot is None:
            return True
        return datetime.utcnow() - self.last_snapshot.timestamp >= self.refresh_interval

    def current_universe(self, fallback: List[str]) -> List[str]:
        if self.last_snapshot and self.last_snapshot.eligible:
            return self.last_snapshot.eligible
        return fallback
