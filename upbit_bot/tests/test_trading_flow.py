import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from upbit_bot.app import OrderPlan, TradingBot
from upbit_bot.storage import SQLiteStateStore
from upbit_bot.strategy.composite import Decision, Signal
from upbit_bot.trading.account import AccountSnapshot, Holding
from upbit_bot.trading.executor import OrderChance, place_order, validate_order


def test_validate_order_enforces_minimum_total():
    constraints = OrderChance(
        bid_fee=0.0005,
        ask_fee=0.0005,
        bid_min_total=50000.0,
        ask_min_total=60000.0,
        max_total=1_000_000.0,
        tick_size=1.0,
    )
    result = validate_order(
        side="bid",
        price=10_000.0,
        volume=2.0,
        constraints=constraints,
        simulated=True,
    )
    assert not result.ok
    assert "최소금액" in (result.rejection_reason or "")


def test_calculate_volume_respects_exchange_minimum_and_risk():
    bot = TradingBot(markets=["KRW-BTC"], simulated=True, use_ai=False)
    bot.per_trade_risk_pct = 100.0
    bot.price_buffer.append("KRW-BTC", 5_000.0)
    bot.account_snapshot = AccountSnapshot(krw_balance=100_000.0, holdings=[], total_value=100_000.0)
    decision = Decision(market="KRW-BTC", price=5_000.0, score=50.0, signal=Signal.BUY, reason="test")
    constraints = OrderChance(
        bid_fee=0.0005,
        ask_fee=0.0005,
        bid_min_total=10_000.0,
        ask_min_total=10_000.0,
        max_total=0.0,
        tick_size=1.0,
    )

    plan: OrderPlan = bot._calculate_volume(decision, constraints)
    assert plan.rejection_reason is None
    assert plan.volume > 0
    assert math.isclose(plan.price, 5_000.0)
    assert plan.price * plan.volume >= 30_000.0


def _mock_response(volume):
    resp = MagicMock()
    resp.ok = True
    resp.text = ""
    resp.json.return_value = {"executed_volume": volume}
    return resp


def test_place_order_handles_partial_fill_and_retry():
    with patch("upbit_bot.trading.executor.requests.post") as mock_post:
        mock_post.side_effect = [_mock_response(0.5), _mock_response(0.5)]
        result = place_order(
            access_key="A",
            secret_key="B",
            market="KRW-BTC",
            side="ask",
            volume=1.0,
            price=10_000.0,
            simulated=False,
            validation_logs=["사전 검증 완료"],
        )

    assert result.success
    assert math.isclose(result.volume, 1.0)
    assert len(result.raw.get("attempts", [])) == 2
    assert any("부분 체결" in log for log in result.validation_logs)


def test_state_store_persists_snapshot_and_positions(tmp_path):
    store = SQLiteStateStore(db_path=tmp_path / "state.db")
    store.ensure_schema()
    holdings = [
        Holding(market="KRW-BTC", currency="BTC", balance=0.5, avg_buy_price=10_000.0, estimated_krw=5_000.0)
    ]
    snapshot = AccountSnapshot(
        krw_balance=100_000.0,
        holdings=holdings,
        total_value=105_000.0,
        total_fee=500.0,
        profit=5_000.0,
        profit_pct=5.0,
    )

    store.persist_snapshot(snapshot)
    positions = store.load_positions()

    assert len(positions) == 1
    assert positions[0].market == "KRW-BTC"
    assert math.isclose(positions[0].avg_price, 10_000.0)
