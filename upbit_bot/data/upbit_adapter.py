"""Upbit 웹소켓 우선 어댑터.

- 모든 시세/호가 요청을 웹소켓 단발 구독으로 대체
- REST 호출은 웹소켓 실패 시에만 보조적으로 사용
- Remaining-Req 헤더 기반 레이트리밋 파싱 및 최소 대기
"""
from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import asyncio
import requests
import websockets

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upbit.com/v1"
WS_URL = "wss://api.upbit.com/websocket/v1"


@dataclass
class RateLimit:
    group: str
    min_remaining: int
    sec_remaining: int

    @classmethod
    def parse(cls, header: str | None) -> Optional["RateLimit"]:
        if not header:
            return None
        try:
            parts = {}
            for item in header.split(";"):
                if "=" not in item:
                    continue
                k, v = item.split("=", 1)
                parts[k.strip()] = v.strip()
            return cls(
                group=parts.get("group", ""),
                min_remaining=int(parts.get("min", "0")),
                sec_remaining=int(parts.get("sec", "0")),
            )
        except Exception:
            logger.debug("Remaining-Req 파싱 실패: %s", header, exc_info=True)
            return None


class UpbitAdapter:
    def __init__(self, *, timeout: int = 10, max_retries: int = 3, backoff: float = 0.6) -> None:
        self._session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    async def _collect_ws_messages(self, payload: list[dict], *, expected: int, timeout: float = 5.0) -> List[dict]:
        """웹소켓 단발 구독으로 원하는 개수만큼 메시지를 모은다."""

        results: List[dict] = []
        try:
            async with websockets.connect(WS_URL, ping_interval=30) as ws:
                await ws.send(json.dumps(payload))
                while len(results) < expected:
                    remaining = timeout if timeout > 0 else None
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    data = json.loads(msg)
                    if isinstance(data, dict):
                        results.append(data)
        except Exception:
            logger.exception("웹소켓 스냅샷 수집 실패")
        return results

    def _run_ws(self, coro: asyncio.Future) -> List[dict]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            logger.debug("실행 중인 이벤트 루프에서는 동기 웹소켓 스냅샷을 건너뜁니다.")
            return []

    def _wait_rate_limit(self, headers: Dict[str, str]) -> None:
        rl = RateLimit.parse(headers.get("Remaining-Req"))
        if not rl:
            return
        # 초당 한도 여유가 적을 때만 최소 슬립으로 보호
        if rl.sec_remaining <= 1:
            time.sleep(self.backoff)

    def request(self, method: str, path: str, *, params: Optional[dict] = None) -> Any:
        url = f"{BASE_URL}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(method, url, params=params, timeout=self.timeout)
                self._wait_rate_limit(resp.headers)
                if resp.status_code == 429:
                    sleep_for = self.backoff * attempt
                    logger.warning("429 응답. %.2fs 대기 후 재시도", sleep_for)
                    time.sleep(sleep_for)
                    continue
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} 서버 오류", response=resp)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: PERF203 (재시도 루프에서 필요)
                last_exc = exc
                sleep_for = self.backoff * attempt
                logger.warning("요청 실패(%s). %.2fs 후 재시도 (%d/%d)", exc, sleep_for, attempt, self.max_retries)
                time.sleep(sleep_for)
        if last_exc:
            raise last_exc
        raise RuntimeError("요청 실패")

    def list_markets(self, *, is_details: bool = False) -> List[dict]:
        params = {"isDetails": str(is_details).lower()}
        # 마켓 목록은 웹소켓에서 제공하지 않아 REST를 최소 범위로 유지
        data = self.request("GET", "/market/all", params=params)
        return list(data)

    def ticker(self, markets: Iterable[str]) -> List[dict]:
        markets_list = list(markets)
        payload = [
            {"ticket": "snapshot"},
            {"type": "ticker", "codes": markets_list, "isOnlyRealtime": True},
        ]
        ws_result = self._run_ws(self._collect_ws_messages(payload, expected=len(markets_list)))
        if ws_result:
            return ws_result
        params = {"markets": ",".join(markets_list)}
        return list(self.request("GET", "/ticker", params=params))

    def candles(self, *, market: str, kind: str, unit: int, count: int) -> List[dict]:
        # 캔들 API는 웹소켓에 직접 대응이 없으므로 거래 스트림 1회 수집으로 보조
        # (초기 구간 채우기용 최소 구현, 실패 시 REST 보조)
        payload = [
            {"ticket": "trade"},
            {"type": "trade", "codes": [market], "isOnlyRealtime": True},
        ]
        ws_trades = self._run_ws(self._collect_ws_messages(payload, expected=max(1, min(count, 50))))
        if ws_trades:
            normalized: List[dict] = []
            for trade in ws_trades:
                ts = trade.get("trade_timestamp") or trade.get("timestamp")
                if not ts:
                    continue
                normalized.append(
                    {
                        "candle_date_time_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%S", time.gmtime(ts / 1000)
                        ),
                        "opening_price": float(trade.get("trade_price", 0.0)),
                        "high_price": float(trade.get("trade_price", 0.0)),
                        "low_price": float(trade.get("trade_price", 0.0)),
                        "trade_price": float(trade.get("trade_price", 0.0)),
                        "candle_acc_trade_volume": float(trade.get("trade_volume", 0.0)),
                    }
                )
            return normalized

        path = f"/candles/{kind}/{unit}" if kind == "minutes" else "/candles/days"
        params = {"market": market, "count": count}
        data = self.request("GET", path, params=params)
        return list(data)

    def trades(self, *, market: str, count: int = 50) -> List[dict]:
        payload = [
            {"ticket": "trade"},
            {"type": "trade", "codes": [market], "isOnlyRealtime": True},
        ]
        ws_data = self._run_ws(self._collect_ws_messages(payload, expected=count))
        if ws_data:
            return ws_data

        params = {"market": market, "count": count}
        data = self.request("GET", "/trades/ticks", params=params)
        return list(data)
