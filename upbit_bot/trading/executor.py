"""
주문 실행기. 실제 주문은 환경설정에 따라 모의모드/실거래모드로 동작하도록 설계한다.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    side: str
    market: str
    volume: float
    price: float
    raw: dict
    fee: float
    net_amount: float
    fee_rate: float
    error: Optional[str] = None
    validation_logs: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


@dataclass
class OrderChance:
    bid_fee: float
    ask_fee: float
    bid_min_total: float
    ask_min_total: float
    max_total: float
    tick_size: float


@dataclass
class ValidationResult:
    ok: bool
    logs: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


def _jwt_token(access_key: str, secret_key: str, query: dict) -> str:
    payload = {"access_key": access_key, "nonce": str(int(time.time() * 1000))}
    if query:
        q = json.dumps(query, separators=(",", ":"), ensure_ascii=False)
        m = hashlib.sha512()
        m.update(q.encode())
        payload["query_hash"] = m.hexdigest()
        payload["query_hash_alg"] = "SHA512"

    header = {"typ": "JWT", "alg": "HS256"}

    def _b64(obj: dict) -> str:
        encoded = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(encoded).decode().rstrip("=")

    header_b64 = _b64(header)
    payload_b64 = _b64(payload)
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def _parse_remaining_req(header_value: str) -> Dict[str, int]:
    tokens = [t.strip() for t in header_value.split(";") if "=" in t]
    result: Dict[str, int] = {}
    for token in tokens:
        key, value = token.split("=", 1)
        try:
            result[key] = int(value)
        except ValueError:
            continue
    return result


def _compute_throttle_delay(headers: Mapping[str, str]) -> float:
    remaining_header = headers.get("Remaining-Req") if headers else None
    if not remaining_header:
        return 0.0

    parsed = _parse_remaining_req(remaining_header)
    sec_remaining = parsed.get("sec")
    min_remaining = parsed.get("min")
    delay = 0.0

    if sec_remaining is not None and sec_remaining <= 1:
        delay = max(delay, 1.0)
    if min_remaining is not None and min_remaining <= 1:
        delay = max(delay, 1.0)

    return delay


def _respect_rate_limit(headers: Mapping[str, str]) -> None:
    delay = _compute_throttle_delay(headers)
    if delay > 0:
        logger.debug("Remaining-Req 헤더 기준으로 %.2fs 대기", delay)
        time.sleep(delay)


def _request_with_retry(
    method: str,
    url: str,
    *,
    request_fn: Optional[Callable[..., requests.Response]] = None,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> Optional[requests.Response]:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            if request_fn:
                response = request_fn(
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=timeout,
                )
            else:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=timeout,
                )
        except Exception:
            logger.exception("%s 요청 중 네트워크 오류 (attempt=%d)", method, attempt)
            if attempt == max_retries:
                return None
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue

        _respect_rate_limit(response.headers)
        if response.status_code in (429, 500, 502, 503, 504):
            logger.warning(
                "요청 실패(code=%s), %.1fs 후 재시도 (attempt=%d/%d)",
                response.status_code,
                backoff,
                attempt,
                max_retries,
            )
            if attempt == max_retries:
                return response
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
            continue

        return response

    return None


def _infer_tick_size(price: float) -> float:
    if price >= 2_000_000:
        return 1000.0
    if price >= 1_000_000:
        return 500.0
    if price >= 500_000:
        return 100.0
    if price >= 100_000:
        return 50.0
    if price >= 10_000:
        return 10.0
    if price >= 1_000:
        return 5.0
    if price >= 100:
        return 1.0
    if price >= 10:
        return 0.1
    if price >= 1:
        return 0.01
    return 0.001


def _normalize_price(price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return price
    normalized = round(price / tick_size) * tick_size
    return round(normalized, 8)


def _normalize_volume(volume: float) -> float:
    return round(max(volume, 0.0), 8)


def fetch_order_chance(*, access_key: str, secret_key: str, market: str) -> Optional[OrderChance]:
    if not access_key or not secret_key:
        logger.debug("API 키가 없어 주문 가능 정보를 조회하지 않습니다.")
        return None

    query = {"market": market}
    jwt = _jwt_token(access_key, secret_key, query)
    headers = {"Authorization": f"Bearer {jwt}"}
    resp = _request_with_retry(
        "GET",
        "https://api.upbit.com/v1/orders/chance",
        request_fn=requests.get,
        headers=headers,
        params=query,
    )
    if resp is None:
        logger.error("주문 가능 정보 조회에 반복 실패")
        return None

    if not resp.ok:
        logger.error("주문 가능 정보 응답 오류: %s", resp.text)
        return None

    data = resp.json()
    market_info = data.get("market", {}) if isinstance(data, dict) else {}
    bid_info = market_info.get("bid", {})
    ask_info = market_info.get("ask", {})
    return OrderChance(
        bid_fee=float(data.get("bid_fee", 0.0)),
        ask_fee=float(data.get("ask_fee", 0.0)),
        bid_min_total=float(bid_info.get("min_total", 0.0) or 0.0),
        ask_min_total=float(ask_info.get("min_total", 0.0) or 0.0),
        max_total=float(market_info.get("max_total", 0.0) or 0.0),
        tick_size=float(bid_info.get("price_unit") or ask_info.get("price_unit") or 0.0),
    )


def validate_order(
    *,
    side: str,
    price: float,
    volume: float,
    constraints: Optional[OrderChance],
    simulated: bool,
) -> ValidationResult:
    logs: List[str] = []
    if price <= 0 or volume <= 0:
        reason = "가격 또는 수량이 0 이하입니다."
        return ValidationResult(False, logs, reason)

    notional = price * volume
    if constraints:
        tick = constraints.tick_size or _infer_tick_size(price)
        normalized_price = _normalize_price(price, tick)
        if abs(normalized_price - price) > 1e-6:
            logs.append(f"호가 단위에 맞게 가격을 {normalized_price}로 보정 필요")
        min_total = constraints.bid_min_total if side == "bid" else constraints.ask_min_total
        max_total = constraints.max_total or float("inf")
        if notional < min_total:
            reason = f"주문 금액이 거래소 최소금액 {min_total:.0f}원 미만입니다."
            return ValidationResult(False, logs, reason)
        if notional > max_total:
            reason = f"주문 금액이 거래소 최대금액 {max_total:.0f}원을 초과합니다."
            return ValidationResult(False, logs, reason)
    elif not simulated and not constraints:
        logs.append("거래소 제약 정보를 가져오지 못해 검증을 완화합니다.")

    return ValidationResult(True, logs, None)


def place_order(
    *,
    access_key: str,
    secret_key: str,
    market: str,
    side: str,
    volume: float,
    price: Optional[float] = None,
    simulated: bool = True,
    fee_rate: float = 0.0005,
    validation_logs: Optional[List[str]] = None,
    allow_retry: bool = True,
) -> OrderResult:
    """
    업비트 주문 API 호출 혹은 모의주문 수행.
    """
    logger.info("주문 요청: %s %s %.4f @ %s (simulated=%s)", side, market, volume, price, simulated)
    amount = (price or 0.0) * volume
    fee = amount * fee_rate
    net_amount = amount - fee
    logs = list(validation_logs or [])
    if simulated:
        return OrderResult(
            True,
            side,
            market,
            volume,
            price or 0.0,
            raw={"simulated": True},
            fee=fee,
            net_amount=net_amount,
            fee_rate=fee_rate,
            validation_logs=logs,
        )

    url = "https://api.upbit.com/v1/orders"
    body = {"market": market, "side": side, "volume": str(volume), "ord_type": "limit"}
    if price:
        body["price"] = str(price)

    jwt = _jwt_token(access_key, secret_key, body)
    headers = {"Authorization": f"Bearer {jwt}"}

    response = _request_with_retry("POST", url, request_fn=requests.post, headers=headers, json_body=body)
    if response is None:
        return OrderResult(
            False,
            side,
            market,
            volume,
            price or 0.0,
            raw={},
            fee=fee,
            net_amount=net_amount,
            fee_rate=fee_rate,
            error="주문 API 반복 실패",
            validation_logs=logs,
            rejection_reason="주문 API 반복 실패",
        )

    if not response.ok:
        logger.error("주문 실패: %s", response.text)
        return OrderResult(
            False,
            side,
            market,
            volume,
            price or 0.0,
            raw={},
            fee=fee,
            net_amount=net_amount,
            fee_rate=fee_rate,
            error=response.text,
            validation_logs=logs,
            rejection_reason="주문 API 오류",
        )

    raw_response = response.json()
    executed_volume = float(raw_response.get("executed_volume", 0.0) or 0.0)
    filled_volume = executed_volume if executed_volume > 0 else volume
    attempts = [raw_response]

    if allow_retry and executed_volume < volume:
        remaining = max(volume - executed_volume, 0.0)
        if remaining > 0:
            logs.append(f"부분 체결 발생({executed_volume:.8f}/{volume:.8f}), 시장가 재시도")
            market_body = {"market": market, "side": side, "ord_type": "market"}
            if side == "bid":
                market_body["price"] = str((price or 0.0) * remaining)
            else:
                market_body["volume"] = str(remaining)
            retry_resp = _request_with_retry(
                "POST", url, request_fn=requests.post, headers=headers, json_body=market_body, max_retries=2
            )
            if retry_resp is None:
                return OrderResult(
                    False,
                    side,
                    market,
                    executed_volume,
                    price or 0.0,
                    raw={"attempts": attempts},
                    fee=executed_volume * (price or 0.0) * fee_rate,
                    net_amount=executed_volume * (price or 0.0) * (1 - fee_rate),
                    fee_rate=fee_rate,
                    error="시장가 재시도 중 네트워크 오류",
                    validation_logs=logs,
                )
            if retry_resp.ok:
                retry_json = retry_resp.json()
                attempts.append(retry_json)
                retry_filled = float(retry_json.get("executed_volume", remaining) or remaining)
                filled_volume += retry_filled
            else:
                attempts.append({"error": retry_resp.text})
                logger.error("시장가 재시도 실패: %s", retry_resp.text)

    filled_volume = min(filled_volume, volume)
    executed_amount = (price or 0.0) * filled_volume
    fee = executed_amount * fee_rate
    net_amount = executed_amount - fee

    return OrderResult(
        True,
        side,
        market,
        filled_volume,
        price or 0.0,
        raw={"attempts": attempts},
        fee=fee,
        net_amount=net_amount,
        fee_rate=fee_rate,
        validation_logs=logs,
    )
