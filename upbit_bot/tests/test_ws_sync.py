import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from upbit_bot.data.stream import PriceBuffer
from upbit_bot.storage import SQLiteStateStore
from upbit_bot.trading.ws_sync import UpbitWebSocketSync


@pytest.mark.asyncio
async def test_ws_sync_updates_price_buffer_and_persists_account(tmp_path):
    store = SQLiteStateStore(db_path=tmp_path / "state.db")
    store.ensure_schema()
    buffer = PriceBuffer()
    sync = UpbitWebSocketSync(
        markets=["KRW-BTC"],
        price_buffer=buffer,
        state_store=store,
        access_key="",
        secret_key="",
    )

    # mock websockets.connect to feed a single ticker message then stop
    fake_ws = AsyncMock()
    fake_ws.__aenter__.return_value = fake_ws
    fake_ws.__aexit__.return_value = False
    fake_ws.recv = AsyncMock()

    async def fake_iter():
        yield json.dumps({"code": "KRW-BTC", "trade_price": 1000.0})
        raise asyncio.CancelledError()

    fake_ws.__aiter__.side_effect = fake_iter

    stop = asyncio.Event()
    with patch("upbit_bot.trading.ws_sync.websockets.connect", return_value=fake_ws):
        task = asyncio.create_task(sync._stream_public(stop_event=stop))
        await asyncio.sleep(0.01)
        stop.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert buffer.latest("KRW-BTC") == pytest.approx(1000.0)
