import types

import pytest
import requests

from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.data.upbit_adapter import UpbitAdapter


class DummyResponse:
    def __init__(self, status: int, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"status {self.status_code}")

    def json(self):
        return self._payload

    @property
    def ok(self):
        return self.status_code < 400


class DummyAdapter(UpbitAdapter):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.deadlines = []

    def list_markets(self, *, is_details: bool = False, deadline=None):  # noqa: D401
        self.calls.append("list")
        self.deadlines.append(("list", deadline))
        return [
            {"market": "KRW-BTC"},
            {"market": "KRW-ETH"},
            {"market": "USDT-XRP"},
        ]

    def ticker(self, markets, *, deadline=None):  # noqa: D401
        self.calls.append("ticker")
        self.deadlines.append(("ticker", deadline))
        return [
            {"market": "KRW-BTC", "acc_trade_price_24h": 200},
            {"market": "KRW-ETH", "acc_trade_price_24h": 100},
            {"market": "USDT-XRP", "acc_trade_price_24h": 50},
        ]


def test_adapter_retries_and_rate_limit(monkeypatch):
    adapter = UpbitAdapter(max_retries=2, backoff=0)
    responses = [
        DummyResponse(500, {}),
        DummyResponse(200, {"ok": True}, headers={"Remaining-Req": "group=market; min=1; sec=0"}),
    ]
    sleep_calls = []

    def fake_request(method, url, params=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(adapter, "_session", types.SimpleNamespace(request=fake_request))
    monkeypatch.setattr("upbit_bot.data.upbit_adapter.time.sleep", lambda s: sleep_calls.append(s))

    result = adapter.request("GET", "/ticker", params={"markets": "KRW-BTC"})
    assert result == {"ok": True}
    # 500 -> retry -> success
    assert len(sleep_calls) >= 1


def test_fetch_markets_uses_adapter_and_filters(monkeypatch):
    dummy = DummyAdapter()
    markets = fetch_markets(is_fiat=True, fiat_symbol="KRW", top_by_volume=1, adapter=dummy)
    assert markets == ["KRW-BTC"]
    assert dummy.calls == ["list", "ticker"]


def test_fetch_markets_limits_even_when_volume_fails():
    class FaultyAdapter(DummyAdapter):
        def ticker(self, markets, *, deadline=None):  # noqa: D401
            raise RuntimeError("boom")

    dummy = FaultyAdapter()
    markets = fetch_markets(is_fiat=True, fiat_symbol="KRW", top_by_volume=2, adapter=dummy)
    # 거래대금 조회가 실패해도 상위 N개 제한 로직은 유지되어야 한다.
    assert markets == ["KRW-BTC", "KRW-ETH"]


def test_fetch_markets_time_budget_is_optional(monkeypatch):
    dummy = DummyAdapter()
    monkeypatch.setattr("upbit_bot.data.market_fetcher.time.monotonic", lambda: 100.0)

    markets = fetch_markets(
        is_fiat=True, fiat_symbol="KRW", top_by_volume=2, time_budget=None, adapter=dummy
    )

    assert markets == ["KRW-BTC", "KRW-ETH"]
    # 시간 제한을 전달하지 않으면 deadline이 걸리지 않는다.
    assert dummy.deadlines == [("list", None), ("ticker", None)]


def test_request_honors_deadline(monkeypatch):
    adapter = UpbitAdapter(max_retries=5, backoff=1, timeout=5)
    calls = []

    def fake_monotonic():
        fake_monotonic.t += 0.04
        return fake_monotonic.t

    fake_monotonic.t = 0.0
    monkeypatch.setattr("upbit_bot.data.upbit_adapter.time.monotonic", fake_monotonic)
    monkeypatch.setattr("upbit_bot.data.upbit_adapter.time.sleep", lambda s: calls.append(("sleep", s)))

    def fake_request(method, url, params=None, timeout=None):
        calls.append(("timeout", timeout))
        raise requests.Timeout("slow")

    adapter._session = types.SimpleNamespace(request=fake_request)

    with pytest.raises(TimeoutError):
        adapter.request("GET", "/ticker", params={}, deadline=0.12)

    # deadline이 가까우면 긴 백오프 없이 빠르게 중단해야 한다.
    sleep_durations = [dur for kind, dur in calls if kind == "sleep"]
    assert sum(sleep_durations) < 0.2
