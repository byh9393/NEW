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
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.data.stream import PriceBuffer, run_stream
from upbit_bot.strategy.composite import Decision, Signal, evaluate
from upbit_bot.strategy.openai_assistant import AIDecision, evaluate_with_openai
from upbit_bot.trading.executor import OrderResult, place_order

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

    async def start(self) -> None:
        logger.info("거래봇 시작. 모니터링 시장 수: %d", len(self.markets))
        stream_task = asyncio.create_task(run_stream(self.markets, self.price_buffer, stop_event=self.stop_event))
        try:
            while not self.stop_event.is_set():
                await self._tick()
                await asyncio.sleep(1)
        finally:
            self.stop_event.set()
            await stream_task

    async def _tick(self) -> None:
        for market in self.markets:
            prices = self.price_buffer.get_prices(market)
            if len(prices) < 50:
                continue
            series = pd.Series(prices)
            base_decision = evaluate(market, series)
            decision = base_decision
            ai_raw = ""

            if self.use_ai:
                ai_decision: AIDecision = evaluate_with_openai(
                    market,
                    series,
                    api_key=self._openai_key,
                    model=self.openai_model,
                    base_decision=base_decision,
                )
                decision = ai_decision.decision
                ai_raw = ai_decision.raw_response

            executed = False
            order_result: Optional[OrderResult] = None
            if decision.signal != Signal.HOLD:
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
                )
                try:
                    self.on_update(update)
                except Exception:
                    logger.exception("on_update 콜백 처리 중 오류")

    def execute(self, decision: Decision) -> OrderResult:
        side = "bid" if decision.signal == Signal.BUY else "ask"
        volume = max(0.0005, 10000 / decision.price)  # 간단한 고정 리스크 배팅
        access = os.environ.get("UPBIT_ACCESS_KEY", "")
        secret = os.environ.get("UPBIT_SECRET_KEY", "")
        return place_order(
            access_key=access,
            secret_key=secret,
            market=decision.market,
            side=side,
            volume=volume,
            price=decision.price,
            simulated=self.simulated or not access or not secret,
        )

    def stop(self) -> None:
        self.stop_event.set()


async def main() -> None:
    markets = fetch_markets(is_fiat=True, fiat_symbol="KRW")
    bot = TradingBot(markets=markets, simulated=True)
    try:
        await asyncio.wait_for(bot.start(), timeout=10)
    except asyncio.TimeoutError:
        logger.info("데모 타임아웃으로 봇을 종료합니다.")


if __name__ == "__main__":
    asyncio.run(main())
