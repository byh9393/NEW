"""Upbit 시장 목록을 가져오고 필터링하는 유틸리티."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Sequence

from upbit_bot.data.upbit_adapter import UpbitAdapter

logger = logging.getLogger(__name__)


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
            tickers = adapter.ticker(batch)
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


def fetch_markets(
    *,
    is_fiat: bool = True,
    fiat_symbol: str = "KRW",
    top_by_volume: int | None = None,
    adapter: UpbitAdapter | None = None,
) -> List[str]:
    """업비트의 거래가능 시장 목록을 조회한다."""

    client = adapter or UpbitAdapter()
    markets: Sequence[dict] = client.list_markets(is_details=False)

    if not is_fiat:
        filtered = [market["market"] for market in markets]
    else:
        filtered = [market["market"] for market in markets if market["market"].startswith(fiat_symbol)]

    logger.info("가져온 시장 수: %d", len(filtered))

    if top_by_volume:
        deadline = time.monotonic() + 8
        try:
            volumes = _fetch_24h_volumes(filtered, client, deadline=deadline)
        except Exception:
            logger.exception("거래대금 순위 조회 중 오류. 상위 %d개로만 제한합니다.", top_by_volume)
            volumes = {}

        filtered = sorted(filtered, key=lambda m: volumes.get(m, 0.0), reverse=True)
        filtered = filtered[:top_by_volume]
        logger.info("24시간 거래대금 상위 %d개 시장으로 제한", len(filtered))

    return filtered
