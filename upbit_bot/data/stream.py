"""
Upbit 웹소켓으로 시세를 스트리밍하여 버퍼에 저장한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import AsyncIterator, Deque, Dict, Iterable, List, MutableMapping, Optional

import websockets

logger = logging.getLogger(__name__)

UPBIT_WS_ENDPOINT = "wss://api.upbit.com/websocket/v1"


class PriceBuffer:
    """최근 틱 가격을 시장별로 보관하는 순환 버퍼."""

    def __init__(self, maxlen: int = 300):
        self._data: MutableMapping[str, Deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))

    def append(self, market: str, price: float) -> None:
        self._data[market].append(price)
        logger.debug("%s 가격 추가: %.2f (총 %d개)", market, price, len(self._data[market]))

    def get_prices(self, market: str) -> List[float]:
        return list(self._data.get(market, []))

    def latest(self, market: str) -> Optional[float]:
        buffer = self._data.get(market)
        if buffer:
            return buffer[-1]
        return None


async def stream_ticker(markets: Iterable[str]) -> AsyncIterator[dict]:
    """
    업비트 웹소켓 ticker 스트림을 비동기로 생성한다.

    Args:
        markets: 구독할 마켓 리스트.

    Yields:
        수신된 ticker json 딕셔너리.
    """
    subscribe_payload = [
        {"ticket": "tickers"},
        {"type": "ticker", "codes": list(markets), "isOnlyRealtime": True},
    ]
    async with websockets.connect(UPBIT_WS_ENDPOINT, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_payload))
        logger.info("웹소켓 구독 시작: %d개 마켓", len(list(markets)))
        async for message in websocket:
            yield json.loads(message)


async def run_stream(markets: Iterable[str], buffer: PriceBuffer, *, stop_event: asyncio.Event) -> None:
    """
    웹소켓 스트림을 실행하여 버퍼에 가격을 저장한다.

    stop_event가 set 되면 종료한다.
    """
    try:
        async for payload in stream_ticker(markets):
            market = payload.get("code")
            trade_price = payload.get("trade_price")
            if market and trade_price:
                buffer.append(market, float(trade_price))
            if stop_event.is_set():
                break
    except Exception:
        logger.exception("웹소켓 스트림 오류")
        raise
    finally:
        logger.info("스트림 종료")
