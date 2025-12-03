"""
REST 캔들 폴링과 웹소켓 틱 스트림을 결합해 다중 타임프레임 캔들을 관리한다.

- 1m/3m/5m/15m/1h/4h/1d 캔들을 rolling 윈도우로 유지
- 누락된 캔들 간격, 급등락(spike) 감지 및 로깅
- 로컬 캐시(JSON)를 이용해 재시작 시 마지막 타임스탬프 이후 백필 수행
- 웹소켓 재연결/백오프 및 구독 재시도, 중단 시 REST 백필
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import websockets

from upbit_bot.data.stream import PriceBuffer

logger = logging.getLogger(__name__)

UPBIT_WS_ENDPOINT = "wss://api.upbit.com/websocket/v1"
BASE_REST_URL = "https://api.upbit.com/v1/candles"

# 타임프레임 문자열 -> (kind, unit, seconds)
_TIMEFRAME_MAP: Dict[str, Tuple[str, int, int]] = {
    "1m": ("minutes", 1, 60),
    "3m": ("minutes", 3, 180),
    "5m": ("minutes", 5, 300),
    "15m": ("minutes", 15, 900),
    "1h": ("minutes", 60, 3600),
    "4h": ("minutes", 240, 14400),
    "1d": ("days", 1, 86400),
}


@dataclass
class Candle:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candle":
        return cls(
            start=datetime.fromisoformat(data["start"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
        )


class OhlcvService:
    """웹소켓 틱과 REST 캔들 백필을 결합한 OHLCV 관리 서비스."""

    def __init__(
        self,
        markets: Iterable[str],
        *,
        price_buffer: Optional[PriceBuffer] = None,
        timeframes: Iterable[str] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d"),
        window: int = 800,
        cache_dir: str | Path = "./.cache/ohlcv",
    ) -> None:
        self.markets = list(markets)
        self.price_buffer = price_buffer or PriceBuffer(maxlen=1000)
        self.timeframes = tuple(tf for tf in timeframes if tf in _TIMEFRAME_MAP)
        self.window = window
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._candles: Dict[str, Dict[str, Deque[Candle]]] = defaultdict(
            lambda: {tf: deque(maxlen=self.window) for tf in self.timeframes}
        )
        self._stop_event = asyncio.Event()

    async def run(self, *, stop_event: asyncio.Event) -> None:
        """웹소켓/REST 백그라운드 태스크를 실행한다."""

        self._stop_event = stop_event
        await self._load_cache()
        await self._initial_sync()

        ws_task = asyncio.create_task(self._websocket_loop())
        rest_task = asyncio.create_task(self._rest_poll_loop())

        await stop_event.wait()
        ws_task.cancel()
        rest_task.cancel()
        try:
            await asyncio.gather(ws_task, rest_task)
        except Exception:
            logger.debug("종료 중 태스크 취소", exc_info=True)

        await self._persist_cache()

    def get_series(self, market: str, timeframe: str) -> pd.Series:
        candles = list(self._candles.get(market, {}).get(timeframe, []))
        if not candles:
            return pd.Series(dtype=float)
        closes = [c.close for c in candles]
        return pd.Series(closes, index=[c.start for c in candles])

    def get_multi_series(self, market: str, timeframes: Iterable[str]) -> Dict[str, pd.Series]:
        return {tf: self.get_series(market, tf) for tf in timeframes if tf in self.timeframes}

    def get_frame(self, market: str, timeframe: str) -> pd.DataFrame:
        candles = list(self._candles.get(market, {}).get(timeframe, []))
        if not candles:
            return pd.DataFrame()
        data = {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
        return pd.DataFrame(data, index=[c.start for c in candles])

    def get_multi_frames(self, market: str, timeframes: Iterable[str]) -> Dict[str, pd.DataFrame]:
        return {tf: self.get_frame(market, tf) for tf in timeframes if tf in self.timeframes}

    async def _websocket_loop(self) -> None:
        backoff = 1
        subscribe_payload = [
            {"ticket": "ohlcv"},
            {"type": "ticker", "codes": self.markets, "isOnlyRealtime": True},
        ]
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(UPBIT_WS_ENDPOINT, ping_interval=30) as websocket:
                    await websocket.send(json.dumps(subscribe_payload))
                    logger.info("OHLCV 웹소켓 구독 시작 (%d개 시장)", len(self.markets))
                    backoff = 1
                    async for message in websocket:
                        payload = json.loads(message)
                        market = payload.get("code")
                        price = payload.get("trade_price")
                        volume = payload.get("trade_volume", 0.0)
                        ts_ms = payload.get("trade_timestamp") or payload.get("timestamp")
                        if market and price and ts_ms:
                            ts = datetime.utcfromtimestamp(ts_ms / 1000)
                            self.price_buffer.append(market, float(price))
                            self._update_with_tick(market, float(price), float(volume), ts)
                        if self._stop_event.is_set():
                            break
            except Exception:
                logger.exception("웹소켓 스트림 오류, 재시도합니다")
                await self._backfill_all()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _rest_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._sync_all_from_rest()
            except Exception:
                logger.exception("REST 폴링 중 오류")
            await asyncio.sleep(30)

    async def _initial_sync(self) -> None:
        try:
            await self._sync_all_from_rest(full_window=True)
        except Exception:
            logger.exception("초기 REST 동기화 실패")

    async def _sync_all_from_rest(self, *, full_window: bool = False) -> None:
        for market in self.markets:
            for timeframe in self.timeframes:
                await asyncio.to_thread(self._sync_market_timeframe, market, timeframe, full_window)

    def _sync_market_timeframe(self, market: str, timeframe: str, full_window: bool = False) -> None:
        candles = self._fetch_rest_candles(market, timeframe, count=self.window if full_window else 50)
        if not candles:
            return
        existing = self._candles[market][timeframe]
        latest_existing = existing[-1].start if existing else None
        for candle in candles:
            if latest_existing and candle.start <= latest_existing:
                # 이미 존재하는 최신 구간 이후만 추가
                continue
            self._append_candle(market, timeframe, candle)

    def _update_with_tick(self, market: str, price: float, volume: float, ts: datetime) -> None:
        for timeframe in self.timeframes:
            bucket_start = self._bucket_start(ts, timeframe)
            deque_candles = self._candles[market][timeframe]
            if deque_candles and deque_candles[-1].start == bucket_start:
                candle = deque_candles[-1]
                candle.high = max(candle.high, price)
                candle.low = min(candle.low, price)
                candle.close = price
                candle.volume += volume
            else:
                new_candle = Candle(
                    start=bucket_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                )
                self._append_candle(market, timeframe, new_candle)

    def _append_candle(self, market: str, timeframe: str, candle: Candle) -> None:
        deque_candles = self._candles[market][timeframe]
        if deque_candles and candle.start <= deque_candles[-1].start:
            # 중복 또는 과거 데이터는 무시
            if candle.start == deque_candles[-1].start:
                deque_candles[-1] = candle
            return

        if deque_candles:
            prev = deque_candles[-1]
            expected_delta = timedelta(seconds=_TIMEFRAME_MAP[timeframe][2])
            gap = candle.start - prev.start
            if gap > expected_delta * 1.5:
                logger.warning(
                    "캔들 누락 감지: %s %s %s 이후 %s까지 공백", market, timeframe, prev.start, candle.start
                )
            self._detect_spike(market, timeframe, prev, candle)
        deque_candles.append(candle)

    def _detect_spike(self, market: str, timeframe: str, prev: Candle, current: Candle) -> None:
        if prev.close <= 0:
            return
        change = abs(current.close - prev.close) / prev.close
        if change >= 0.05:
            logger.warning(
                "가격 스파이크 감지: %s %s %.2f -> %.2f (%.2f%%)",
                market,
                timeframe,
                prev.close,
                current.close,
                change * 100,
            )

    def _fetch_rest_candles(self, market: str, timeframe: str, *, count: int = 200) -> List[Candle]:
        kind, unit, _ = _TIMEFRAME_MAP[timeframe]
        url = f"{BASE_REST_URL}/{kind}/{unit}" if kind == "minutes" else f"{BASE_REST_URL}/{kind}"
        params = {"market": market, "count": min(count, 200)}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        raw: List[dict] = response.json()
        candles: List[Candle] = []
        for item in raw:
            start_str = item.get("candle_date_time_utc") or item.get("candle_date_time_kst")
            if not start_str:
                continue
            start = datetime.fromisoformat(start_str)
            candle = Candle(
                start=start,
                open=float(item.get("opening_price", 0.0)),
                high=float(item.get("high_price", 0.0)),
                low=float(item.get("low_price", 0.0)),
                close=float(item.get("trade_price", 0.0)),
                volume=float(item.get("candle_acc_trade_volume", 0.0)),
            )
            candles.append(candle)
        candles.sort(key=lambda c: c.start)
        return candles

    def _bucket_start(self, ts: datetime, timeframe: str) -> datetime:
        _, unit, _ = _TIMEFRAME_MAP[timeframe]
        if timeframe.endswith("d"):
            return datetime(ts.year, ts.month, ts.day)
        minute_block = (ts.minute // unit) * unit
        return datetime(ts.year, ts.month, ts.day, ts.hour, minute_block)

    async def _persist_cache(self) -> None:
        tasks = [asyncio.to_thread(self._persist_market_cache, market) for market in self.markets]
        await asyncio.gather(*tasks)

    def _persist_market_cache(self, market: str) -> None:
        data = {
            timeframe: [c.to_dict() for c in candles]
            for timeframe, candles in self._candles.get(market, {}).items()
        }
        path = self.cache_dir / f"{market.replace('/', '_')}.json"
        path.write_text(json.dumps(data, ensure_ascii=False))

    async def _load_cache(self) -> None:
        tasks = [asyncio.to_thread(self._load_market_cache, market) for market in self.markets]
        await asyncio.gather(*tasks)

    def _load_market_cache(self, market: str) -> None:
        path = self.cache_dir / f"{market.replace('/', '_')}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            logger.exception("캔들 캐시 로드 실패: %s", path)
            return
        for timeframe, candles in data.items():
            if timeframe not in self.timeframes:
                continue
            dq = deque(maxlen=self.window)
            for raw in candles:
                dq.append(Candle.from_dict(raw))
            self._candles[market][timeframe] = dq

    async def _backfill_all(self) -> None:
        for market in self.markets:
            for timeframe in self.timeframes:
                await asyncio.to_thread(self._backfill_market_timeframe, market, timeframe)

    def _backfill_market_timeframe(self, market: str, timeframe: str) -> None:
        existing = self._candles[market][timeframe]
        latest = existing[-1].start if existing else None
        candles = self._fetch_rest_candles(market, timeframe, count=200)
        if not candles:
            return
        if latest:
            candles = [c for c in candles if c.start > latest]
        for candle in candles:
            self._append_candle(market, timeframe, candle)

