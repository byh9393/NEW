"""Upbit 시장 목록을 가져오고 필터링하는 유틸리티."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

from upbit_bot.data.upbit_adapter import UpbitAdapter

logger = logging.getLogger(__name__)


def _fetch_24h_volumes(markets: Sequence[str], adapter: UpbitAdapter) -> Dict[str, float]:
    """요청한 마켓들의 24시간 거래대금을 반환한다."""

    volumes: Dict[str, float] = {}
    for idx in range(0, len(markets), 100):
        batch = markets[idx : idx + 100]
        tickers = adapter.ticker(batch)
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
        try:
            volumes = _fetch_24h_volumes(filtered, client)
            filtered = sorted(filtered, key=lambda m: volumes.get(m, 0.0), reverse=True)
            filtered = filtered[:top_by_volume]
            logger.info("24시간 거래대금 상위 %d개 시장으로 제한", len(filtered))
        except Exception:
            logger.exception("거래대금 순위 조회 중 오류. 전체 목록을 반환합니다.")

    return filtered
