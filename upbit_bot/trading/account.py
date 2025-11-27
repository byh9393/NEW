"""
업비트 계좌 정보 조회 및 스냅샷 모델.

- 실거래 모드: 업비트 계좌 API를 통해 보유 자산과 원화 잔고를 조회
- 모의 모드: 트레이딩 봇 내부 상태를 사용해 스냅샷을 구성
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import requests

from upbit_bot.trading.executor import _jwt_token

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    market: str
    currency: str
    balance: float
    avg_buy_price: float
    estimated_krw: float


@dataclass
class AccountSnapshot:
    krw_balance: float
    holdings: List[Holding]
    total_value: float


def fetch_account_snapshot(
    *,
    access_key: str,
    secret_key: str,
    price_lookup: Callable[[str], Optional[float]],
) -> Optional[AccountSnapshot]:
    """업비트 계좌 API를 호출하여 최신 잔고를 반환한다."""
    if not access_key or not secret_key:
        logger.warning("업비트 API 키가 설정되지 않아 계좌를 조회할 수 없습니다.")
        return None

    headers = {"Authorization": f"Bearer {_jwt_token(access_key, secret_key, {})}"}
    try:
        resp = requests.get("https://api.upbit.com/v1/accounts", headers=headers, timeout=10)
    except Exception:
        logger.exception("계좌 조회 중 네트워크 오류")
        return None

    if not resp.ok:
        logger.error("계좌 조회 실패: %s", resp.text)
        return None

    holdings: List[Holding] = []
    krw_balance = 0.0
    total_value = 0.0
    for item in resp.json():
        currency = item.get("currency")
        balance = float(item.get("balance", 0.0))
        avg_price = float(item.get("avg_buy_price", 0.0))
        if currency == "KRW":
            krw_balance = balance
            total_value += krw_balance
            continue
        market = f"KRW-{currency}" if currency else ""
        last_price = price_lookup(market) if market else None
        estimated = balance * (last_price if last_price else avg_price)
        total_value += estimated
        holdings.append(
            Holding(
                market=market,
                currency=currency or "",
                balance=balance,
                avg_buy_price=avg_price,
                estimated_krw=estimated,
            )
        )

    return AccountSnapshot(krw_balance=krw_balance, holdings=holdings, total_value=total_value)
