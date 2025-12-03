"""
포지션·포트폴리오 리스크 관리 모듈.

- 거래당 위험 한도, 종목/포트폴리오 최대 비중, 일일 손실 한도, 최대 드로우다운 감지
- 상관성 기반 동시 진입 제한 및 하루·종목당 진입 횟수 제한
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

from upbit_bot.trading.account import AccountSnapshot


@dataclass
class RiskLimits:
    per_trade_risk_pct: float
    max_position_pct: float
    max_portfolio_pct: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    correlation_threshold: float
    max_correlated_positions: int
    max_entries_per_day: int
    max_entries_per_symbol: int
    universe_top_n: int


class RiskPortfolioManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.daily_entry_count = 0
        self.symbol_entry_count: Dict[str, int] = {}
        self._entry_day = date.today()
        self.halted = False

    def reset_if_new_day(self) -> None:
        today = date.today()
        if today != self._entry_day:
            self._entry_day = today
            self.daily_entry_count = 0
            self.symbol_entry_count.clear()

    def register_entry(self, market: str) -> None:
        self.reset_if_new_day()
        self.daily_entry_count += 1
        self.symbol_entry_count[market] = self.symbol_entry_count.get(market, 0) + 1

    def register_exit(self, market: str) -> None:
        if market in self.symbol_entry_count:
            self.symbol_entry_count[market] = max(0, self.symbol_entry_count[market] - 1)

    def check_drawdown(self, snapshot: Optional[AccountSnapshot]) -> Optional[str]:
        if not snapshot or snapshot.total_value <= 0:
            return None
        if snapshot.profit_pct <= -self.limits.max_drawdown_pct:
            self.halted = True
            return f"최대 드로우다운 {self.limits.max_drawdown_pct:.1f}% 도달로 전략 정지"
        return None

    def _exposure_limits(self, snapshot: Optional[AccountSnapshot], market: str) -> Optional[str]:
        if not snapshot or snapshot.total_value <= 0:
            return None
        total_value = snapshot.total_value
        total_positions_value = sum(h.estimated_krw for h in snapshot.holdings)
        if total_positions_value / total_value * 100 > self.limits.max_portfolio_pct:
            return f"포트폴리오 총노출 {total_positions_value/total_value*100:.1f}%>한도 {self.limits.max_portfolio_pct:.1f}%"
        for holding in snapshot.holdings:
            weight = holding.estimated_krw / total_value * 100
            if holding.market == market and weight > self.limits.max_position_pct:
                return f"종목 비중 {weight:.1f}%가 한도 {self.limits.max_position_pct:.1f}% 초과"
        return None

    def _entry_limits(self, market: str) -> Optional[str]:
        self.reset_if_new_day()
        if self.daily_entry_count >= self.limits.max_entries_per_day:
            return "일일 진입 한도 초과"
        if self.symbol_entry_count.get(market, 0) >= self.limits.max_entries_per_symbol:
            return "종목별 진입 한도 초과"
        return None

    def _correlation_guard(self, correlated_positions: int) -> Optional[str]:
        if correlated_positions >= self.limits.max_correlated_positions:
            return f"상관성 높은 포지션 {correlated_positions}개 보유로 동시 진입 제한"
        return None

    def validate_entry(
        self,
        *,
        market: str,
        snapshot: Optional[AccountSnapshot],
        correlated_positions: int,
        universe_rank: Optional[int],
    ) -> Optional[str]:
        if self.halted:
            return "전략이 드로우다운으로 중지됨"

        drawdown_reason = self.check_drawdown(snapshot)
        if drawdown_reason:
            return drawdown_reason

        if snapshot and snapshot.profit_pct <= -self.limits.daily_loss_limit_pct:
            return f"일 손실 {snapshot.profit_pct:.2f}%가 한도 {self.limits.daily_loss_limit_pct:.1f}% 초과"

        exposure_reason = self._exposure_limits(snapshot, market)
        if exposure_reason:
            return exposure_reason

        entry_reason = self._entry_limits(market)
        if entry_reason:
            return entry_reason

        corr_reason = self._correlation_guard(correlated_positions)
        if corr_reason:
            return corr_reason

        if universe_rank and universe_rank > self.limits.universe_top_n:
            return f"거래대금 상위 {self.limits.universe_top_n}위 밖 (순위 {universe_rank})"

        return None
