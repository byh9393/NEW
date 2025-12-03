"""
백테스트 & 리서치 엔진.

- 업비트 OHLCV 데이터와 실시간 전략 로직(Decision/evaluate, RiskManager 등)을 공유해 일관된 시뮬레이션을 수행한다.
- 수수료·슬리피지·거래소 최소 주문금액(봇 내부 30,000원 기준)과 호가 반올림을 반영한다.
- 포트폴리오 단위 지표(총수익률, 연율화 수익률, 승률/Profit Factor, MDD, Sharpe/Sortino, 평균 보유기간)를 제공한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from upbit_bot.strategy.composite import Decision, Signal, evaluate
from upbit_bot.trading.exit_rules import evaluate_exit
from upbit_bot.trading.executor import _infer_tick_size, _normalize_price


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.08  # 0.08% 기본 슬리피지
    min_order_krw: float = 30_000.0
    risk_per_trade_pct: float = 2.0
    daily_loss_limit_pct: float = 5.0
    target_timeframe: str = "5m"


@dataclass
class Trade:
    market: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    pnl_pct: float
    reason: str

    @property
    def holding_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        *,
        evaluator=evaluate,
        exit_evaluator=evaluate_exit,
    ) -> None:
        self.config = config or BacktestConfig()
        self.evaluator = evaluator
        self.exit_evaluator = exit_evaluator

    def run(self, data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Args:
            data: {market: DataFrame(OHLCV)} 형태. 최소 `close` 컬럼 필요.
        """
        cash = self.config.initial_cash
        positions: Dict[str, Dict[str, float | datetime]] = {}
        equity_history: List[float] = []
        time_index = self._merge_time_index(data.values())
        trades: List[Trade] = []
        start_date = None
        daily_start_value = cash

        for ts in time_index:
            # 데일리 손실 한도 초기화
            if start_date is None or ts.date() != start_date:
                start_date = ts.date()
                daily_start_value = cash + sum(
                    pos["volume"] * pos["last_price"] for pos in positions.values()
                )

            for market, frame in data.items():
                if self.config.target_timeframe == "1d":
                    frame = frame.resample("1D").agg("last").dropna()
                if ts not in frame.index:
                    continue
                prices = frame.loc[:ts, "close"]
                if prices.size < 60:
                    continue
                last_price = float(prices.iloc[-1])

                # 보유 포지션 업데이트
                if market in positions:
                    positions[market]["last_price"] = last_price
                    entry_price = positions[market]["entry_price"]
                    exit_signal = self.exit_evaluator(
                        prices, entry_price=entry_price, entry_time=positions[market]["entry_time"]
                    )
                    if exit_signal.should_exit:
                        trade, cash = self._close_position(
                            market=market,
                            ts=ts,
                            price=last_price,
                            positions=positions,
                            cash=cash,
                            reason=exit_signal.reason,
                        )
                        trades.append(trade)
                        continue

                decision: Decision = self.evaluator(market, prices)
                if decision.signal == Signal.BUY and market not in positions:
                    if self._daily_loss_exceeded(cash, positions, daily_start_value):
                        continue
                    filled = self._open_position(market, ts, last_price, decision.score, cash)
                    if filled:
                        positions[market] = filled["position"]
                        cash = filled["cash"]
                elif decision.signal == Signal.SELL and market in positions:
                    trade, cash = self._close_position(
                        market=market,
                        ts=ts,
                        price=last_price,
                        positions=positions,
                        cash=cash,
                        reason=decision.reason,
                    )
                    trades.append(trade)

            equity = cash + sum(pos["volume"] * pos["last_price"] for pos in positions.values())
            equity_history.append((ts, equity))

        # 종료 시점에 남은 포지션을 청산하여 누락된 손익을 반영한다.
        if positions and time_index:
            last_ts = time_index[-1]
            for market, pos in list(positions.items()):
                trade, cash = self._close_position(
                    market=market,
                    ts=last_ts,
                    price=pos["last_price"],
                    positions=positions,
                    cash=cash,
                    reason="세션 종료 청산",
                )
                trades.append(trade)
            equity_history.append((last_ts, cash))

        equity_series = pd.Series({ts: eq for ts, eq in equity_history}).sort_index()
        stats = self._calc_stats(trades, equity_series)
        return BacktestResult(trades=trades, equity_curve=equity_series, stats=stats)

    def _merge_time_index(self, frames: Iterable[pd.DataFrame]) -> List[datetime]:
        merged = set()
        for frame in frames:
            merged.update(frame.index)
        return sorted(merged)

    def _daily_loss_exceeded(self, cash: float, positions: Dict[str, Dict[str, float]], start_value: float) -> bool:
        current = cash + sum(pos["volume"] * pos["last_price"] for pos in positions.values())
        drop_pct = (current - start_value) / start_value * 100 if start_value > 0 else 0
        return drop_pct <= -self.config.daily_loss_limit_pct

    def _open_position(
        self, market: str, ts: datetime, price: float, score: float, cash: float
    ) -> Optional[Dict[str, object]]:
        tick = _infer_tick_size(price)
        entry_price = _normalize_price(price * (1 + self.config.slippage_pct / 100), tick)
        risk_cap = cash * (self.config.risk_per_trade_pct / 100)
        stop_distance = max(price * 0.012, price * 0.015)  # 최소 1.2~1.5% 손절폭
        risk_volume = risk_cap / stop_distance
        min_volume = self.config.min_order_krw / entry_price
        affordable_volume = cash / (entry_price * (1 + self.config.fee_rate))
        volume = min(max(risk_volume, min_volume), affordable_volume)
        notional = entry_price * volume
        fee = notional * self.config.fee_rate

        if volume <= 0 or notional < self.config.min_order_krw:
            return None

        cash -= notional + fee
        position = {
            "entry_price": entry_price,
            "volume": volume,
            "entry_time": ts,
            "last_price": price,
            "entry_fee": fee,
            "score": score,
        }
        return {"position": position, "cash": cash}

    def _close_position(
        self,
        *,
        market: str,
        ts: datetime,
        price: float,
        positions: Dict[str, Dict[str, float]],
        cash: float,
        reason: str,
    ) -> tuple[Trade, float]:
        pos = positions.pop(market)
        exit_price = _normalize_price(price * (1 - self.config.slippage_pct / 100), _infer_tick_size(price))
        volume = pos["volume"]
        notional = exit_price * volume
        exit_fee = notional * self.config.fee_rate
        pnl = notional - pos["entry_price"] * volume - pos.get("entry_fee", 0.0) - exit_fee
        pnl_pct = pnl / (pos["entry_price"] * volume + pos.get("entry_fee", 0.0)) * 100 if pos["entry_price"] > 0 else 0
        cash = cash + notional - exit_fee
        trade = Trade(
            market=market,
            entry_time=pos["entry_time"],
            exit_time=ts,
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            volume=volume,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )
        return trade, cash

    def _calc_stats(self, trades: List[Trade], equity: pd.Series) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if equity.empty:
            return stats
        initial = equity.iloc[0]
        final = equity.iloc[-1]
        total_return = (final - initial) / initial * 100 if initial else 0
        days = max((equity.index[-1] - equity.index[0]).days, 1)
        annualized = ((final / initial) ** (365 / days) - 1) * 100 if initial > 0 else 0
        returns = equity.pct_change().dropna()
        downside = returns[returns < 0]
        sharpe = returns.mean() / (returns.std() + 1e-9) * math.sqrt(365 * 24 * 12)
        sortino = returns.mean() / (downside.std() + 1e-9) * math.sqrt(365 * 24 * 12)
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        mdd = drawdown.min() * 100

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))) if losses else float("inf")
        avg_hold = float(np.mean([t.holding_minutes for t in trades])) if trades else 0

        stats.update(
            {
                "total_return_pct": total_return,
                "annualized_return_pct": annualized,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown_pct": float(mdd),
                "sharpe": float(sharpe),
                "sortino": float(sortino),
                "avg_holding_minutes": avg_hold,
                "trades": len(trades),
            }
        )
        return stats

    
