"""Upbit REST 어댑터.

- /v1/* 호출 공통 래퍼
- Remaining-Req 헤더 기반 레이트리밋 파싱 및 최소 대기
- 429/5xx 재시도와 Gzip 대응
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upbit.com/v1"


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
        data = self.request("GET", "/market/all", params=params)
        return list(data)

    def ticker(self, markets: Iterable[str]) -> List[dict]:
        params = {"markets": ",".join(markets)}
        return list(self.request("GET", "/ticker", params=params))

    def candles(self, *, market: str, kind: str, unit: int, count: int) -> List[dict]:
        path = f"/candles/{kind}/{unit}" if kind == "minutes" else "/candles/days"
        params = {"market": market, "count": count}
        data = self.request("GET", path, params=params)
        return list(data)

    def trades(self, *, market: str, count: int = 50) -> List[dict]:
        params = {"market": market, "count": count}
        data = self.request("GET", "/trades/ticks", params=params)
        return list(data)
