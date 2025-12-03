import types

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

    def list_markets(self, *, is_details: bool = False):  # noqa: D401
        self.calls.append("list")
        return [
            {"market": "KRW-BTC"},
            {"market": "KRW-ETH"},
            {"market": "USDT-XRP"},
        ]

    def ticker(self, markets):  # noqa: D401
        self.calls.append("ticker")
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
