"""
WebSocket synchronization helpers for live ticker and private order streams.

- Public: subscribes to ticker streams and updates the shared PriceBuffer.
- Private: subscribes to myOrder stream (if API keys provided) and triggers account refresh + persistence.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable, Optional, Callable

import websockets

from upbit_bot.data.stream import UPBIT_WS_ENDPOINT, PriceBuffer
from upbit_bot.trading.account import fetch_account_snapshot, AccountSnapshot
from upbit_bot.trading.executor import _jwt_token
from upbit_bot.storage import SQLiteStateStore

logger = logging.getLogger(__name__)


class UpbitWebSocketSync:
    def __init__(
        self,
        *,
        markets: Iterable[str],
        price_buffer: PriceBuffer,
        state_store: SQLiteStateStore,
        access_key: str = "",
        secret_key: str = "",
        on_account: Optional[Callable[[AccountSnapshot], None]] = None,
    ) -> None:
        self.markets = list(markets)
        self.price_buffer = price_buffer
        self.state_store = state_store
        self.access_key = access_key
        self.secret_key = secret_key
        self.on_account = on_account

    async def _stream_public(self, stop_event: asyncio.Event) -> None:
        payload = [
            {"ticket": "public"},
            {"type": "ticker", "codes": self.markets, "isOnlyRealtime": True},
        ]
        backoff = 1
        while not stop_event.is_set():
            try:
                async with websockets.connect(UPBIT_WS_ENDPOINT, ping_interval=60) as ws:
                    await ws.send(json.dumps(payload))
                    async for msg in ws:
                        data = json.loads(msg)
                        code = data.get("code")
                        price = data.get("trade_price")
                        if code and price:
                            self.price_buffer.append(code, float(price))
                        if stop_event.is_set():
                            break
                backoff = 1
            except Exception:
                logger.exception("공개 웹소켓 스트림 오류, 재연결합니다")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _stream_private(self, stop_event: asyncio.Event) -> None:
        if not self.access_key or not self.secret_key:
            return
        token = _jwt_token(self.access_key, self.secret_key, {})
        headers = {"Authorization": f"Bearer {token}"}
        payload = [
            {"ticket": "private"},
            {"type": "myOrder", "codes": self.markets},
            {"format": "SIMPLE"},
        ]
        backoff = 1
        while not stop_event.is_set():
            try:
                async with websockets.connect(UPBIT_WS_ENDPOINT, ping_interval=60, extra_headers=headers) as ws:
                    await ws.send(json.dumps(payload))
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            data = None

                        # parse minimal order deltas
                        if isinstance(data, dict):
                            event = data.get("event") or data.get("state")
                            if event in ("order", "trade", "done", "watch"):
                                logger.info("myOrder event: %s %s %s", event, data.get("side"), data.get("code"))
                                try:
                                    self.state_store.record_myorder_event(data)
                                except Exception:
                                    logger.exception("record_myorder_event failed")

                        snapshot = fetch_account_snapshot(
                            access_key=self.access_key,
                            secret_key=self.secret_key,
                            price_lookup=self.price_buffer.latest,
                        )
                        if snapshot:
                            snapshot.total_fee = snapshot.total_fee or 0.0
                            if self.on_account:
                                try:
                                    self.on_account(snapshot)
                                except Exception:
                                    logger.exception("on_account callback failed")
                            try:
                                self.state_store.persist_snapshot(snapshot)
                            except Exception:
                                logger.exception("persist_snapshot failed")
                        if stop_event.is_set():
                            break
                backoff = 1
            except Exception:
                logger.exception("개인 웹소켓 스트림 오류, 재연결합니다")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def run(self, stop_event: asyncio.Event) -> None:
        tasks = [asyncio.create_task(self._stream_public(stop_event))]
        if self.access_key and self.secret_key:
            tasks.append(asyncio.create_task(self._stream_private(stop_event)))
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
