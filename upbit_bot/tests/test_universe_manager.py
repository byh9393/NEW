import pandas as pd

from upbit_bot.data.universe import UniverseManager


def _frame(turnovers):
    idx = pd.date_range("2024-01-01", periods=len(turnovers), freq="D")
    closes = [1.0] * len(turnovers)
    return pd.DataFrame({"close": closes, "volume": turnovers}, index=idx)


def test_universe_filters_by_30d_turnover_and_spread():
    manager = UniverseManager(min_30d_avg_turnover=100.0, max_spread_pct=0.02, top_n=2)
    frames = {
        "KRW-A": _frame([120] * 35),
        "KRW-B": _frame([150] * 35),
        "KRW-C": _frame([80] * 35),
        "KRW-D": _frame([130] * 35),
    }
    spreads = {"KRW-A": 0.01, "KRW-B": 0.03, "KRW-D": 0.015}

    snapshot = manager.refresh(markets=list(frames.keys()), frame_lookup=frames, spreads=spreads)

    # KRW-C는 평균 거래대금 미달, KRW-B는 스프레드 초과로 제외
    assert snapshot.eligible == ["KRW-D", "KRW-A"]
    assert snapshot.turnover_30d_avg["KRW-A"] >= 120
    assert snapshot.turnover_24h["KRW-A"] == 120


def test_universe_should_refresh_interval():
    manager = UniverseManager(min_30d_avg_turnover=10, refresh_interval=pd.Timedelta(seconds=0))
    frames = {"KRW-A": _frame([20] * 35)}
    manager.refresh(markets=["KRW-A"], frame_lookup=frames)
    assert manager.should_refresh() is True
