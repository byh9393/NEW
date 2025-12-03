import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from upbit_bot.backtest import BacktestConfig, BacktestEngine
from upbit_bot.strategy.composite import Decision, Signal


def _sample_frame(length: int, start_price: float, drift: float) -> pd.DataFrame:
    ts = pd.date_range(datetime(2024, 1, 1), periods=length, freq="5min")
    steps = np.arange(length)
    prices = start_price + drift * steps + np.sin(steps / 5) * 3
    volume = np.full(length, 100.0)
    return pd.DataFrame({"close": prices, "volume": volume}, index=ts)


def test_backtest_engine_runs_and_reports_stats():
    frame_a = _sample_frame(180, 100.0, 0.6)
    frame_b = _sample_frame(180, 50.0, -0.2) + 1.0  # 약세 흐름

    def trending_evaluator(market, prices):
        if len(prices) < 20:
            return Decision(market, float(prices.iloc[-1]), 0.0, Signal.HOLD, "데이터 부족")
        recent = prices.iloc[-1]
        prev = prices.iloc[-6]
        if recent > prev * 1.01:
            return Decision(market, float(recent), 80.0, Signal.BUY, "상승 추세")
        if recent < prev * 0.995:
            return Decision(market, float(recent), -60.0, Signal.SELL, "약세 전환")
        return Decision(market, float(recent), 0.0, Signal.HOLD, "중립")

    engine = BacktestEngine(
        BacktestConfig(initial_cash=200_000.0, slippage_pct=0.05), evaluator=trending_evaluator
    )
    result = engine.run({"KRW-AAA": frame_a, "KRW-BBB": frame_b})

    assert result.stats
    assert "total_return_pct" in result.stats
    assert isinstance(result.equity_curve, pd.Series)
    assert len(result.equity_curve) > 0
    # 일부 포지션 청산이 이뤄져야 한다.
    assert len(result.trades) > 0
    assert not math.isnan(result.stats.get("max_drawdown_pct", 0.0))
