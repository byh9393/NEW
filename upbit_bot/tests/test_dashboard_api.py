import json

import pytest
from fastapi.testclient import TestClient

from upbit_bot.storage import SQLiteStateStore
from upbit_bot.trading.account import AccountSnapshot, Holding
from upbit_bot.ui.api import get_app


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "state.db"
    store = SQLiteStateStore(db_path=db)
    store.ensure_schema()
    holdings = [
        Holding(market="KRW-BTC", currency="BTC", balance=0.1, avg_buy_price=10_000_000.0, estimated_krw=1_000_000.0)
    ]
    snap = AccountSnapshot(
        krw_balance=2_000_000.0,
        holdings=holdings,
        total_value=3_000_000.0,
        total_fee=10_000.0,
        profit=100_000.0,
        profit_pct=3.4,
    )
    store.persist_snapshot(snap)
    return store


def test_rest_endpoints_expose_account_and_positions(store):
    app = get_app(store.db_path)
    client = TestClient(app)

    resp = client.get("/account")
    assert resp.status_code == 200
    data = resp.json()["snapshot"]
    assert data["krw_balance"] == pytest.approx(2_000_000.0)
    assert data["holdings"][0]["market"] == "KRW-BTC"

    resp = client.get("/positions")
    assert resp.status_code == 200
    positions = resp.json()["positions"]
    assert positions[0]["market"] == "KRW-BTC"


def test_orders_and_risk_events_return_empty_list(store):
    app = get_app(store.db_path)
    client = TestClient(app)
    assert client.get("/orders").json()["orders"] == []
    assert client.get("/risk-events").json()["events"] == []


def test_websocket_stream_sends_snapshot(store):
    app = get_app(store.db_path)
    client = TestClient(app)
    with client.websocket_connect("/ws/stream") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        assert first["data"]["krw_balance"] == pytest.approx(2_000_000.0)
        ping = ws.receive_json()
        assert ping["type"] == "ping"
