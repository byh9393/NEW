"""
업비트 전 자동매매 오케스트레이터.
비동기 루프를 통해 실시간 가격 수집, 전략 평가, 주문 실행을 관리한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from time import time
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.data.stream import PriceBuffer
from upbit_bot.data.ohlcv_service import OhlcvService
from upbit_bot.strategy.composite import Decision, Signal, evaluate
from upbit_bot.strategy.openai_assistant import AIDecision, evaluate_with_openai
from upbit_bot.trading.account import AccountSnapshot, Holding, fetch_account_snapshot
from upbit_bot.trading.executor import (
    OrderChance,
    OrderResult,
    fetch_order_chance,
    place_order,
    validate_order,
    _infer_tick_size,
    _normalize_price,
    _normalize_volume,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DecisionUpdate:
    market: str
    price: float
    score: float
    signal: Signal
    reason: str
    ai_raw: str
    executed: bool
    order_result: Optional[OrderResult]
    timestamp: datetime
    account: Optional[AccountSnapshot]
    suppress_log: bool


@dataclass
class OrderPlan:
    price: float
    volume: float
    logs: List[str]
    rejection_reason: Optional[str] = None


class TradingBot:
    def __init__(
        self,
        *,
        markets: Iterable[str],
        maxlen: int = 300,
        simulated: bool = True,
        use_ai: bool = True,
        openai_model: str = "gpt-4o-mini",
        on_update: Optional[Callable[[DecisionUpdate], None]] = None,
        fee_rate: float = 0.0005,
    ) -> None:
        self.markets: List[str] = list(markets)
        self.price_buffer = PriceBuffer(maxlen=maxlen)
        self.simulated = simulated
        self.stop_event = asyncio.Event()
        self.positions: Dict[str, float] = {}
        self.use_ai = use_ai
        self.openai_model = openai_model
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.on_update = on_update
        self.krw_balance: float = float(os.environ.get("SIMULATED_CASH", 1_000_000))
        self.avg_price: Dict[str, float] = {}
        self.account_snapshot: Optional[AccountSnapshot] = None
        self._last_account_refresh = 0.0
        self._access = os.environ.get("UPBIT_ACCESS_KEY", "")
        self._secret = os.environ.get("UPBIT_SECRET_KEY", "")
        self.fee_rate = fee_rate
        self.total_fees: float = 0.0
        self.initial_value: Optional[float] = None
        self.max_slippage_pct: float = float(os.environ.get("MAX_SLIPPAGE_PCT", 0.5))
        self.per_trade_risk_pct: float = float(os.environ.get("PER_TRADE_RISK_PCT", 1.0))
        self.daily_loss_limit_pct: float = float(os.environ.get("DAILY_LOSS_LIMIT_PCT", 5.0))
        self.required_timeframes: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.ohlcv_service = OhlcvService(self.markets, price_buffer=self.price_buffer, timeframes=self.required_timeframes)

    async def start(self) -> None:
        logger.info("거래봇 시작. 모니터링 시장 수: %d", len(self.markets))
        ohlcv_task = asyncio.create_task(self.ohlcv_service.run(stop_event=self.stop_event))
        try:
            while not self.stop_event.is_set():
                await self._tick()
                await asyncio.sleep(1)
        finally:
            self.stop_event.set()
            await ohlcv_task

    async def _tick(self) -> None:
        self._refresh_account_snapshot()
        for market in self.markets:
            series_map = self.ohlcv_service.get_multi_series(market, self.required_timeframes)
            primary = series_map.get("5m") or series_map.get("1m") or pd.Series(dtype=float)
            if primary.empty or len(primary) < 50:
                continue
            base_decision = evaluate(market, primary)
            decision = base_decision
            ai_raw = ""

            snapshot = self.account_snapshot
            holdings = {h.market: h.balance for h in snapshot.holdings} if snapshot else self.positions
            has_holding = holdings.get(market, 0.0) > 0
            suppress_due_to_no_holding = False

            if self.use_ai:
                ai_decision: AIDecision = evaluate_with_openai(
                    market,
                    primary,
                    api_key=self._openai_key,
                    model=self.openai_model,
                    base_decision=base_decision,
                )
                decision = ai_decision.decision
                ai_raw = ai_decision.raw_response

            executed = False
            order_result: Optional[OrderResult] = None
            should_execute = decision.signal != Signal.HOLD
            if decision.signal == Signal.SELL and not has_holding:
                should_execute = False
                suppress_due_to_no_holding = True
            elif decision.signal == Signal.HOLD and not has_holding:
                suppress_due_to_no_holding = True

            if should_execute:
                order_result = self.execute(decision)
                executed = True
                logger.info(
                    "%s -> %s (점수 %.1f): %s | AI=%s",
                    market,
                    decision.signal,
                    decision.score,
                    order_result.raw,
                    ai_raw,
                )
                self._refresh_account_snapshot(force=True)

            if self.on_update:
                update = DecisionUpdate(
                    market=market,
                    price=decision.price,
                    score=decision.score,
                    signal=decision.signal,
                    reason=decision.reason,
                    ai_raw=ai_raw,
                    executed=executed,
                    order_result=order_result,
                    timestamp=datetime.utcnow(),
                    account=self.account_snapshot,
                    suppress_log=decision.suppress_log or suppress_due_to_no_holding,
                )
                try:
                    self.on_update(update)
                except Exception:
                    logger.exception("on_update 콜백 처리 중 오류")

    def execute(self, decision: Decision) -> OrderResult:
        side = "bid" if decision.signal == Signal.BUY else "ask"
        constraints = (
            fetch_order_chance(access_key=self._access, secret_key=self._secret, market=decision.market)
            if self._access and self._secret
            else None
        )
        plan = self._calculate_volume(decision, constraints)
        if plan.volume <= 0 or plan.rejection_reason:
            return OrderResult(
                False,
                side,
                decision.market,
                0.0,
                plan.price,
                raw={},
                fee=0.0,
                net_amount=0.0,
                fee_rate=self.fee_rate,
                error=plan.rejection_reason or "주문 조건 불충족",
                validation_logs=plan.logs,
                rejection_reason=plan.rejection_reason or "주문 조건 불충족",
            )

        validation = validate_order(
            side=side,
            price=plan.price,
            volume=plan.volume,
            constraints=constraints,
            simulated=self.simulated or not self._access or not self._secret,
        )
        logs = plan.logs + validation.logs
        if not validation.ok:
            return OrderResult(
                False,
                side,
                decision.market,
                0.0,
                plan.price,
                raw={},
                fee=0.0,
                net_amount=0.0,
                fee_rate=self.fee_rate,
                error=validation.rejection_reason or "주문 조건 불충족",
                validation_logs=logs,
                rejection_reason=validation.rejection_reason or "주문 조건 불충족",
            )

        result = place_order(
            access_key=self._access,
            secret_key=self._secret,
            market=decision.market,
            side=side,
            volume=plan.volume,
            price=plan.price,
            simulated=self.simulated or not self._access or not self._secret,
            fee_rate=self.fee_rate,
            validation_logs=logs,
        )

        if result.success and (self.simulated or not self._access or not self._secret):
            self._apply_simulated_fill(decision, result.volume, price=plan.price, fee_rate=self.fee_rate)

        if result.success and not (self.simulated or not self._access or not self._secret):
            # 실거래 시 주문 응답에 포함된 예상 수수료를 누적 관리
            self.total_fees += result.fee

        return result

    def stop(self) -> None:
        self.stop_event.set()

    def _calculate_volume(self, decision: Decision, constraints: Optional[OrderChance]) -> OrderPlan:
        logs: List[str] = []
        snapshot = self.account_snapshot or AccountSnapshot(
            krw_balance=self.krw_balance, holdings=[], total_value=self.krw_balance
        )
        fee_rate = self.fee_rate
        tick_size = constraints.tick_size if constraints else _infer_tick_size(decision.price)
        price = _normalize_price(decision.price, tick_size)
        slippage_limit = self.max_slippage_pct / 100.0
        last_price = self.price_buffer.latest(decision.market)
        if last_price and last_price > 0:
            slippage = abs(price - last_price) / last_price
            if slippage > slippage_limit:
                reason = f"슬리피지 {slippage*100:.2f}%가 한도 {self.max_slippage_pct:.2f}%를 초과"
                return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)

        if snapshot and snapshot.profit_pct <= -self.daily_loss_limit_pct:
            reason = f"일일 손실 한도 {self.daily_loss_limit_pct:.2f}%를 초과하여 거래 중지"
            return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)

        if constraints:
            min_total_exchange = constraints.bid_min_total if decision.signal == Signal.BUY else constraints.ask_min_total
        else:
            min_total_exchange = 0.0
        min_total = max(30_000.0, min_total_exchange)
        min_volume = min_total / (price * (1 - fee_rate))
        equity = snapshot.total_value if snapshot else self.krw_balance
        risk_cap_amount = equity * (self.per_trade_risk_pct / 100.0)

        if decision.signal == Signal.BUY:
            available_krw = snapshot.krw_balance if snapshot else self.krw_balance
            min_cash_required = min_total / (1 - fee_rate) * (1 + fee_rate)
            if available_krw < min_cash_required:
                reason = f"가용 원화 {available_krw:.0f}원이 최소 요구금액 {min_cash_required:.0f}원 미만"
                return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)
            max_affordable_volume = available_krw / (price * (1 + fee_rate))
            if max_affordable_volume < min_volume:
                reason = "수수료를 고려하면 최소 주문금액을 충족할 수 없습니다."
                return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)
            strength = max(decision.score, 0.0) / 100.0
            weight = 0.05 + (0.25 - 0.05) * strength  # 5~25% 비중 배정
            target_amount = min(available_krw * weight, risk_cap_amount)
            target_volume = target_amount / price
            volume = max(min_volume, target_volume)
            volume = min(max_affordable_volume, volume)
        else:
            holdings = {h.market: h.balance for h in snapshot.holdings} if snapshot else self.positions
            available_volume = holdings.get(decision.market, self.positions.get(decision.market, 0.0))
            if available_volume <= 0:
                reason = f"보유 물량 없음: {decision.market}"
                return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)
            sell_fraction = min(1.0, abs(decision.score) / 70)  # 신호 강도에 따라 최대 전량
            planned_volume = max(0.0, available_volume * sell_fraction)
            if available_volume * price * (1 - fee_rate) < min_total:
                reason = "보유 자산이 최소 주문금액에 미달"
                return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)
            volume = min(available_volume, planned_volume)
            risk_cap_volume = risk_cap_amount / price if risk_cap_amount > 0 else volume
            volume = min(volume, risk_cap_volume)

        volume = _normalize_volume(volume)
        if price * volume < min_total:
            reason = "정규화 후 주문금액이 최소 기준에 미달"
            return OrderPlan(price=price, volume=0.0, logs=[reason], rejection_reason=reason)

        return OrderPlan(price=price, volume=volume, logs=logs)

    def _apply_simulated_fill(self, decision: Decision, volume: float, *, price: float, fee_rate: float) -> None:
        amount = volume * price
        fee = amount * fee_rate
        self.total_fees += fee
        if decision.signal == Signal.BUY:
            acquired_volume = volume * (1 - fee_rate)
            total_cost = amount + fee
            self.krw_balance -= total_cost
            prev_volume = self.positions.get(decision.market, 0.0)
            prev_value = self.avg_price.get(decision.market, 0.0) * prev_volume
            new_volume = prev_volume + acquired_volume
            avg_price = (prev_value + total_cost) / new_volume if new_volume else decision.price
            self.positions[decision.market] = new_volume
            self.avg_price[decision.market] = avg_price
        else:
            proceeds = amount - fee
            self.krw_balance += proceeds
            prev_volume = self.positions.get(decision.market, 0.0)
            remaining = max(0.0, prev_volume - volume)
            if remaining == 0:
                self.positions.pop(decision.market, None)
                self.avg_price.pop(decision.market, None)
            else:
                self.positions[decision.market] = remaining

    def _refresh_account_snapshot(self, *, force: bool = False) -> None:
        now = time()
        if not force and now - self._last_account_refresh < 5:
            return

        if self.simulated or not self._access or not self._secret:
            holdings: List[Holding] = []
            for market, volume in self.positions.items():
                last_price = self.price_buffer.latest(market) or self.avg_price.get(market, 0.0)
                avg_price = self.avg_price.get(market, last_price)
                estimated = volume * (last_price or avg_price or 0.0)
                holdings.append(
                    Holding(
                        market=market,
                        currency=market.split("-")[-1],
                        balance=volume,
                        avg_buy_price=avg_price or 0.0,
                        estimated_krw=estimated,
                    )
                )
            total = self.krw_balance + sum(h.estimated_krw for h in holdings)
            if self.initial_value is None:
                self.initial_value = total
            profit = total - (self.initial_value or total)
            profit_pct = (profit / self.initial_value * 100) if self.initial_value else 0.0
            self.account_snapshot = AccountSnapshot(
                krw_balance=self.krw_balance,
                holdings=holdings,
                total_value=total,
                total_fee=self.total_fees,
                profit=profit,
                profit_pct=profit_pct,
            )
        else:
            snapshot = fetch_account_snapshot(
                access_key=self._access,
                secret_key=self._secret,
                price_lookup=self.price_buffer.latest,
            )
            if snapshot:
                if self.initial_value is None:
                    self.initial_value = snapshot.total_value
                profit = snapshot.total_value - (self.initial_value or snapshot.total_value)
                profit_pct = (profit / self.initial_value * 100) if self.initial_value else 0.0
                snapshot.total_fee = self.total_fees
                snapshot.profit = profit
                snapshot.profit_pct = profit_pct
            self.account_snapshot = snapshot
        self._last_account_refresh = now


async def main() -> None:
    markets = fetch_markets(is_fiat=True, fiat_symbol="KRW", top_by_volume=5)
    bot = TradingBot(markets=markets, simulated=True)
    try:
        await asyncio.wait_for(bot.start(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("데모 타임아웃으로 봇을 종료합니다.")


if __name__ == "__main__":
    asyncio.run(main())
