"""
Upbit 시장 목록을 가져오고 필터링하는 유틸리티.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import requests

UPBIT_MARKETS_URL = "https://api.upbit.com/v1/market/all"
UPBIT_TICKER_URL = "https://api.upbit.com/v1/ticker"

logger = logging.getLogger(__name__)


def _fetch_24h_volumes(markets: Sequence[str]) -> Dict[str, float]:
    """요청한 마켓들의 24시간 거래대금을 반환한다."""

    volumes: Dict[str, float] = {}
    for idx in range(0, len(markets), 100):
        batch = markets[idx : idx + 100]
        params = {"markets": ",".join(batch)}
        response = requests.get(UPBIT_TICKER_URL, params=params, timeout=10)
        response.raise_for_status()
        for ticker in response.json():
            market = ticker.get("market")
            volume = float(ticker.get("acc_trade_price_24h", 0.0))
            if market:
                volumes[market] = volume
    return volumes


def fetch_markets(
    *, is_fiat: bool = True, fiat_symbol: str = "KRW", top_by_volume: int | None = None
) -> List[str]:
    """
    업비트의 거래가능 시장 목록을 조회한다.

    Args:
        is_fiat: 원화마켓만 필터링할지 여부. 기본 True.
        fiat_symbol: 피아트 심볼 (예: "KRW", "BTC").

    Returns:
        시장 심볼 문자열 리스트 (예: ["KRW-BTC", "KRW-ETH"])
    """
    params = {"isDetails": "false"}
    response = requests.get(UPBIT_MARKETS_URL, params=params, timeout=10)
    response.raise_for_status()
    markets: Sequence[dict] = response.json()

    if not is_fiat:
        filtered = [market["market"] for market in markets]
    else:
        filtered = [market["market"] for market in markets if market["market"].startswith(fiat_symbol)]

    logger.info("가져온 시장 수: %d", len(filtered))

    if top_by_volume:
        try:
            volumes = _fetch_24h_volumes(filtered)
            filtered = sorted(filtered, key=lambda m: volumes.get(m, 0.0), reverse=True)
            filtered = filtered[:top_by_volume]
            logger.info("24시간 거래대금 상위 %d개 시장으로 제한", len(filtered))
        except Exception:
            logger.exception("거래대금 순위 조회 중 오류. 전체 목록을 반환합니다.")

    return filtered
