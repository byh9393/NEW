"""
주문 실행기. 실제 주문은 환경설정에 따라 모의모드/실거래모드로 동작하도록 설계한다.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

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


def _jwt_token(access_key: str, secret_key: str, query: dict) -> str:
    payload = {"access_key": access_key, "nonce": str(int(time.time() * 1000))}
    if query:
        q = json.dumps(query, separators=(",", ":"), ensure_ascii=False)
        m = hashlib.sha512()
        m.update(q.encode())
        payload["query_hash"] = m.hexdigest()
        payload["query_hash_alg"] = "SHA512"
    header = {"typ": "JWT", "alg": "HS256"}
    segments = [
        json.dumps(header, separators=(",", ":")).encode(),
        json.dumps(payload, separators=(",", ":")).encode(),
    ]
    signing_input = b".".join(segments)
    signature = hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest()
    return b".".join([signing_input, signature]).decode("latin-1")


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
) -> OrderResult:
    """
    업비트 주문 API 호출 혹은 모의주문 수행.
    """
    logger.info("주문 요청: %s %s %.4f @ %s (simulated=%s)", side, market, volume, price, simulated)
    amount = (price or 0.0) * volume
    fee = amount * fee_rate
    net_amount = amount - fee
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
        )

    url = "https://api.upbit.com/v1/orders"
    body = {"market": market, "side": side, "volume": str(volume), "ord_type": "limit"}
    if price:
        body["price"] = str(price)

    jwt = _jwt_token(access_key, secret_key, body)
    headers = {"Authorization": f"Bearer {jwt}"}

    response = requests.post(url, json=body, headers=headers, timeout=10)
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
        )

    return OrderResult(
        True,
        side,
        market,
        volume,
        price or 0.0,
        raw=response.json(),
        fee=fee,
        net_amount=net_amount,
        fee_rate=fee_rate,
    )
