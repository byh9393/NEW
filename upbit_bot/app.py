"""
업비트 전 자동매매 오케스트레이터.
비동기 루프를 통해 실시간 가격 수집, 전략 평가, 주문 실행을 관리한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.data.stream import PriceBuffer
from upbit_bot.data.ohlcv_service import OhlcvService
from upbit_bot.data.universe import UniverseManager
from upbit_bot.strategy.composite import Decision, Signal, evaluate
from upbit_bot.strategy.openai_assistant import AIDecision, evaluate_with_openai
from upbit_bot.strategy.multiframe import MultiTimeframeAnalyzer, MultiTimeframeFactor
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
from upbit_bot.trading.exit_rules import evaluate_exit, compute_stop_targets
from upbit_bot.trading.risk_portfolio import RiskLimits, RiskPortfolioManager
from upbit_bot.storage import SQLiteStateStore
from upbit_bot.monitoring.alerts import AlertSink, AlertMessage, Severity

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


@dataclass
class EntryMeta:
    entry_time: datetime
    entry_price: float


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
        state_store: Optional[SQLiteStateStore] = None,
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
        self.position_entries: Dict[str, EntryMeta] = {}
        self.max_position_pct = float(os.environ.get("MAX_POSITION_PCT", 25.0))
        self.max_portfolio_pct = float(os.environ.get("MAX_PORTFOLIO_PCT", 90.0))
        self.max_drawdown_pct = float(os.environ.get("MAX_DRAWDOWN_PCT", 12.0))
        self.max_entries_per_day = int(os.environ.get("MAX_ENTRIES_PER_DAY", 12))
        self.max_entries_per_symbol = int(os.environ.get("MAX_ENTRIES_PER_SYMBOL", 3))
        self.universe_top_n = int(os.environ.get("UNIVERSE_TOP_N", 10))
        self.correlation_threshold = float(os.environ.get("CORRELATION_THRESHOLD", 0.8))
        self.max_correlated_positions = int(os.environ.get("MAX_CORRELATED_POSITIONS", 3))
        self.min_30d_turnover = float(os.environ.get("MIN_30D_AVG_TURNOVER", 1_000_000_000))
        self.max_spread_pct = float(os.environ.get("MAX_SPREAD_PCT", 2.0))
        self.universe_refresh_interval = timedelta(
            seconds=int(os.environ.get("UNIVERSE_REFRESH_SEC", 3600))
        )
        self.multi_analyzer = MultiTimeframeAnalyzer(timeframes=["5m", "15m", "1h", "4h", "1d"])
        self.risk_manager = RiskPortfolioManager(
            RiskLimits(
                per_trade_risk_pct=self.per_trade_risk_pct,
                max_position_pct=self.max_position_pct,
                max_portfolio_pct=self.max_portfolio_pct,
                daily_loss_limit_pct=self.daily_loss_limit_pct,
                max_drawdown_pct=self.max_drawdown_pct,
                correlation_threshold=self.correlation_threshold,
                max_correlated_positions=self.max_correlated_positions,
                max_entries_per_day=self.max_entries_per_day,
                max_entries_per_symbol=self.max_entries_per_symbol,
                universe_top_n=self.universe_top_n,
            )
        )
        self.universe_manager = UniverseManager(
            min_30d_avg_turnover=self.min_30d_turnover,
            max_spread_pct=self.max_spread_pct / 100.0,
            top_n=self.universe_top_n,
            refresh_interval=self.universe_refresh_interval,
        )
        self.active_markets: List[str] = list(self.markets)
        self.state_store = state_store or SQLiteStateStore()
        self.state_store.ensure_schema()
        self.alert_sink = AlertSink()
        self._recover_state()

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
        self._refresh_universe()
        correlation_map = self._compute_correlations()
        volume_ranks = self._compute_universe_ranks()
        for market in self.active_markets:
            frames_map = self.ohlcv_service.get_multi_frames(market, self.required_timeframes)
            primary_frame = frames_map.get("5m")
            primary = primary_frame["close"] if primary_frame is not None and not primary_frame.empty else pd.Series(dtype=float)
            if primary.empty or len(primary) < 50:
                continue

            multi_factor = self.multi_analyzer.analyze(
                market,
                frames_map,
                correlation=self._correlation_to_portfolio(market, correlation_map),
            )

            base_decision = evaluate(market, primary)
            decision = base_decision
            ai_raw = ""

            snapshot = self.account_snapshot
            holdings = {h.market: h.balance for h in snapshot.holdings} if snapshot else self.positions
            has_holding = holdings.get(market, 0.0) > 0
            suppress_due_to_no_holding = False

            if has_holding:
                entry_price = self.avg_price.get(market, float(primary.iloc[-1]))
                meta = self.position_entries.get(market)
                exit_signal = evaluate_exit(primary, entry_price=entry_price, entry_time=meta.entry_time if meta else None)
                if exit_signal.should_exit:
                    decision = Decision(
                        market=market,
                        price=float(primary.iloc[-1]),
                        score=decision.score,
                        signal=Signal.SELL,
                        reason=f"리스크 종료: {exit_signal.reason}",
                        quality=decision.quality,
                        suppress_log=False,
                    )

            if self.use_ai:
                ai_decision: AIDecision = evaluate_with_openai(
                    market,
                    primary,
                    api_key=self._openai_key,
                    model=self.openai_model,
                    base_decision=decision,
                )
                decision = ai_decision.decision
                ai_raw = ai_decision.raw_response

            rejection_reason = self._validate_entry_filters(
                market,
                decision,
                multi_factor,
                primary,
                volume_ranks.get(market),
                correlation_map,
            )

            executed = False
            order_result: Optional[OrderResult] = None
            should_execute = decision.signal != Signal.HOLD and not rejection_reason
            if decision.signal == Signal.SELL and not has_holding:
                should_execute = False
                suppress_due_to_no_holding = True
            elif decision.signal == Signal.HOLD and not has_holding:
                suppress_due_to_no_holding = True

            if rejection_reason:
                decision = Decision(
                    market=market,
                    price=float(primary.iloc[-1]),
                    score=decision.score,
                    signal=Signal.HOLD,
                    reason=rejection_reason,
                    quality=decision.quality,
                    suppress_log=True,
                )

            if should_execute:
                order_result = self.execute(decision)
                executed = True
                if decision.signal == Signal.BUY:
                    self.risk_manager.register_entry(market)
                else:
                    self.risk_manager.register_exit(market)
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

        try:
            self.state_store.record_order(order_result=result)
        except Exception:
            logger.exception("주문/체결 기록 실패")

        return result

    def stop(self) -> None:
        self.stop_event.set()

    def _compute_correlations(self, timeframe: str = "1h") -> Dict[str, Dict[str, float]]:
        series_map: Dict[str, pd.Series] = {}
        for market in self.markets:
            series = self.ohlcv_service.get_series(market, timeframe)
            if series is not None and not series.empty:
                series_map[market] = series.pct_change().dropna().tail(120)
        if not series_map:
            return {}
        df = pd.DataFrame(series_map).dropna(how="all")
        if df.empty:
            return {}
        corr = df.corr()
        return {c1: {c2: float(corr.loc[c1, c2]) for c2 in corr.columns if c1 != c2} for c1 in corr.columns}

    def _correlation_to_portfolio(self, market: str, correlation_map: Dict[str, Dict[str, float]]) -> Optional[float]:
        if not correlation_map:
            return None
        holdings = self.account_snapshot.holdings if self.account_snapshot else []
        others = [h.market for h in holdings if h.market != market] if holdings else [m for m in self.positions.keys() if m != market]
        if not others:
            return None
        corr_row = correlation_map.get(market, {})
        values = [abs(corr_row.get(o, 0.0)) for o in others if o in corr_row]
        return float(np.mean(values)) if values else None

    def _count_correlated_positions(self, market: str, correlation_map: Dict[str, Dict[str, float]]) -> int:
        corr_row = correlation_map.get(market, {})
        holdings = self.account_snapshot.holdings if self.account_snapshot else []
        symbols = [h.market for h in holdings] if holdings else list(self.positions.keys())
        return sum(1 for sym in symbols if sym != market and abs(corr_row.get(sym, 0.0)) >= self.correlation_threshold)

    def _compute_universe_ranks(self) -> Dict[str, int]:
        snapshot = self.universe_manager.last_snapshot
        if snapshot:
            ranked = sorted(snapshot.turnover_24h.items(), key=lambda kv: kv[1], reverse=True)
            return {market: idx + 1 for idx, (market, _) in enumerate(ranked)}

        volumes: Dict[str, float] = {}
        for market in self.markets:
            frame = self.ohlcv_service.get_frame(market, "1d")
            if frame is None or frame.empty:
                continue
            latest = frame.iloc[-1]
            turnover = float(latest.get("close", 0.0) * latest.get("volume", 0.0))
            volumes[market] = turnover
        ranked = sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)
        return {market: idx + 1 for idx, (market, _) in enumerate(ranked)}

    def _refresh_universe(self) -> None:
        if not self.universe_manager.should_refresh():
            return
        frames = {m: self.ohlcv_service.get_frame(m, "1d") for m in self.markets}
        snapshot = self.universe_manager.refresh(
            markets=self.markets,
            frame_lookup=frames,
            fetch_spread_if_missing=False,
        )
        self.active_markets = snapshot.eligible or list(self.markets)
        reason = (
            f"유니버스 갱신: {len(self.active_markets)}/{len(self.markets)}개 채택, "
            f"최소 30일 평균 거래대금 {self.min_30d_turnover:,.0f}, 스프레드 {self.max_spread_pct:.2f}%"
        )
        self._notify_risk_event("*", reason)

    def _notify_risk_event(self, market: str, reason: str) -> None:
        try:
            self.state_store.record_risk_event(market=market, reason=reason)
        except Exception:
            logger.exception("리스크 이벤트 기록 실패")
        try:
            self.alert_sink.notify(
                AlertMessage(
                    title=f"리스크 이벤트 - {market}",
                    detail=reason,
                    severity=Severity.WARNING,
                )
            )
        except Exception:
            logger.exception("알림 전송 실패")

    def _validate_entry_filters(
        self,
        market: str,
        decision: Decision,
        multi_factor: MultiTimeframeFactor,
        primary: pd.Series,
        volume_rank: Optional[int],
        correlation_map: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        if decision.signal != Signal.BUY:
            return None

        higher_trend_scores = [multi_factor.trend_by_tf.get(tf, 0.5) for tf in ("1h", "4h", "1d")]
        higher_trend = float(np.mean([s for s in higher_trend_scores if s is not None] or [multi_factor.trend]))
        if higher_trend < 0.55:
            return f"상위 추세 필터 미충족 ({higher_trend:.2f})"

        last_price = float(primary.iloc[-1])
        stop_loss, take_profit = compute_stop_targets(primary, last_price)
        stop_distance = last_price - stop_loss
        target_distance = take_profit - last_price
        if stop_distance <= 0 or target_distance / max(stop_distance, 1e-6) < 1.2:
            return "ATR 대비 목표/손절 리스크-리워드 미충족"

        if volume_rank and volume_rank > self.universe_top_n:
            return f"거래대금 상위 {self.universe_top_n}개 외 종목"

        if multi_factor.composite < 0.52 or multi_factor.momentum < 0.45 or multi_factor.trend < 0.5:
            return (
                f"팩터 엔진 합성점수 부족 (합성 {multi_factor.composite:.2f}/"
                f"추세 {multi_factor.trend:.2f}/모멘텀 {multi_factor.momentum:.2f})"
            )

        correlated_count = self._count_correlated_positions(market, correlation_map)
        risk_reason = self.risk_manager.validate_entry(
            market=market,
            snapshot=self.account_snapshot,
            correlated_positions=correlated_count,
            universe_rank=volume_rank,
        )
        if risk_reason:
            self._notify_risk_event(market, risk_reason)
        return risk_reason

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
            self.position_entries[decision.market] = EntryMeta(entry_time=datetime.utcnow(), entry_price=avg_price)
        else:
            proceeds = amount - fee
            self.krw_balance += proceeds
            prev_volume = self.positions.get(decision.market, 0.0)
            remaining = max(0.0, prev_volume - volume)
            if remaining == 0:
                self.positions.pop(decision.market, None)
                self.avg_price.pop(decision.market, None)
                self.position_entries.pop(decision.market, None)
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
        try:
            if self.account_snapshot:
                self.state_store.persist_snapshot(self.account_snapshot)
        except Exception:
            logger.exception("스냅샷 저장 실패")
        self._last_account_refresh = now

    def _recover_state(self) -> None:
        """로컬 스토어에 저장된 포지션/평단 정보를 활용해 시뮬레이션 세션을 복구한다."""
        if not self.simulated:
            return
        try:
            positions = self.state_store.load_positions()
        except Exception:
            logger.exception("상태 복구 실패")
            return
        for pos in positions:
            self.positions[pos.market] = pos.volume
            self.avg_price[pos.market] = pos.avg_price
            if pos.opened_at:
                self.position_entries[pos.market] = EntryMeta(entry_time=pos.opened_at, entry_price=pos.avg_price)


async def main() -> None:
    markets = fetch_markets(is_fiat=True, fiat_symbol="KRW", top_by_volume=5)
    bot = TradingBot(markets=markets, simulated=True)
    try:
        await asyncio.wait_for(bot.start(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("데모 타임아웃으로 봇을 종료합니다.")


if __name__ == "__main__":
    asyncio.run(main())
