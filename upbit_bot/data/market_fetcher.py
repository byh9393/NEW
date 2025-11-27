"""
Upbit 시장 목록을 가져오고 필터링하는 유틸리티.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import requests

UPBIT_MARKETS_URL = "https://api.upbit.com/v1/market/all"

logger = logging.getLogger(__name__)


def fetch_markets(is_fiat: bool = True, fiat_symbol: str = "KRW") -> List[str]:
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
        return [market["market"] for market in markets]

    filtered = [market["market"] for market in markets if market["market"].startswith(fiat_symbol)]
    logger.info("가져온 시장 수: %d", len(filtered))
    return filtered
