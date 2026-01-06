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

import os

import socket

import requests
from requests.adapters import HTTPAdapter

try:
    # urllib3는 requests의 내부 의존성이다.
    from urllib3.util import connection as urllib3_connection
except Exception:  # pragma: no cover
    urllib3_connection = None

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
    def __init__(
        self,
        *,
        timeout: int = 10,
        max_retries: int = 3,
        backoff: float = 0.6,
        trust_env: bool = False,
        force_ipv4: bool | None = None,
        pool_maxsize: int = 20,
        user_agent: str = "NEW-UpbitBot/1.0",
    ) -> None:
        """Upbit REST 어댑터.

        느린 '마켓 조회'의 가장 흔한 원인은 다음 두 가지다.
        1) 운영체제 프록시/자동 설정(PAC) 탐색으로 인해 requests가 지연됨
        2) IPv6 우선 시도 후 타임아웃까지 기다린 뒤 IPv4로 넘어가며 지연됨

        - trust_env=False: 환경변수 프록시/시스템 프록시 자동 적용을 끈다.
        - force_ipv4=True: (가능한 경우) IPv4만 사용하도록 강제해 초기 연결 지연을 줄인다.
        """

        self._session = requests.Session()
        self._session.trust_env = bool(trust_env)
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip"})

        # 커넥션 풀을 적극적으로 재사용해 TLS 핸드셰이크/연결 비용을 줄인다.
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        self._session.mount("https://", adapter)

        if force_ipv4 is None:
            env = os.environ.get("UPBIT_FORCE_IPV4", "1").strip().lower()
            force_ipv4 = env not in {"0", "false", "no", "off"}

        if force_ipv4 and urllib3_connection is not None:
            # requests -> urllib3의 DNS 해석에서 IPv4만 사용하도록 강제
            try:
                urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
            except Exception:
                logger.debug("IPv4 강제 설정 실패", exc_info=True)

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

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        deadline: float | None = None,
    ) -> Any:
        """Rate-limit과 일시적 장애에 대응하며 REST 호출을 수행한다.

        ``deadline`` 이 주어지면 해당 시점 이전에 호출을 완료하거나 ``TimeoutError``로
        빠르게 중단한다. 마켓 조회 시 UI 제한 시간을 넘겨 실패하는 문제를 방지하기
        위함이다.
        """

        url = f"{BASE_URL}{path}"
        last_exc: Optional[Exception] = None

        for attempt in range(0, self.max_retries + 1):
            if deadline and time.monotonic() >= deadline:
                raise TimeoutError("요청 기한을 초과했습니다.")

            try:
                remaining = None
                if deadline:
                    remaining = max(deadline - time.monotonic(), 0.0)
                    if remaining <= 0:
                        raise TimeoutError("요청 기한을 초과했습니다.")

                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    timeout=min(self.timeout, remaining) if remaining is not None else self.timeout,
                )
                self._wait_rate_limit(resp.headers)

                if resp.status_code == 429:
                    last_exc = requests.HTTPError("429 Too Many Requests", response=resp)
                    if attempt >= self.max_retries:
                        break
                    sleep_for = self.backoff * (attempt + 1)
                    if deadline:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("요청 기한을 초과했습니다.")
                        sleep_for = min(sleep_for, max(remaining, 0))
                    logger.warning("429 응답. %.2fs 대기 후 재시도", sleep_for)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue

                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} 서버 오류", response=resp)

                resp.raise_for_status()
                return resp.json()

            except Exception as exc:  # noqa: PERF203 (재시도 루프에서 필요)
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_for = self.backoff * (attempt + 1)
                if deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("요청 기한을 초과했습니다.")
                    sleep_for = min(sleep_for, max(remaining, 0))
                logger.warning(
                    "요청 실패(%s). %.2fs 후 재시도 (%d/%d)",
                    exc,
                    sleep_for,
                    attempt + 1,
                    self.max_retries,
                )
                if sleep_for > 0:
                    time.sleep(sleep_for)

        if last_exc:
            raise last_exc
        raise RuntimeError("요청 실패")

    def list_markets(self, *, is_details: bool = False, deadline: float | None = None) -> List[dict]:
        params = {"isDetails": str(is_details).lower()}
        data = self.request("GET", "/market/all", params=params, deadline=deadline)
        return list(data)

    def ticker(self, markets: Iterable[str], *, deadline: float | None = None) -> List[dict]:
        params = {"markets": ",".join(markets)}
        return list(self.request("GET", "/ticker", params=params, deadline=deadline))

    def candles(
        self,
        *,
        market: str,
        kind: str,
        unit: int,
        count: int,
        deadline: float | None = None,
    ) -> List[dict]:
        path = f"/candles/{kind}/{unit}" if kind == "minutes" else "/candles/days"
        params = {"market": market, "count": count}
        data = self.request("GET", path, params=params, deadline=deadline)
        return list(data)

    def trades(self, *, market: str, count: int = 50, deadline: float | None = None) -> List[dict]:
        params = {"market": market, "count": count}
        data = self.request("GET", "/trades/ticks", params=params, deadline=deadline)
        return list(data)
