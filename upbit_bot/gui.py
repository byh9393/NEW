"""
거래 진행 상황을 한눈에 볼 수 있는 Tkinter 기반 GUI 대시보드.

- 거래 시작/종료 버튼
- 최근 신호 테이블 및 로그
- 선택한 마켓의 가격 추세 라인 차트

CLI만으로는 어려운 실시간 모니터링을 위해, TradingBot의 on_update 콜백을
사용해 모든 판단/주문 이벤트를 UI에 반영한다.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from upbit_bot.app import DecisionUpdate, TradingBot
from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.trading.account import AccountSnapshot


@dataclass
class BotStatus:
    running: bool = False
    markets: List[str] = field(default_factory=list)
    simulated: bool = True
    use_ai: bool = True


class BotRunner:
    """Tk 루프를 막지 않도록 별도 쓰레드에서 TradingBot을 실행한다."""

    def __init__(self, on_update) -> None:
        self.on_update = on_update
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.bot: Optional[TradingBot] = None

    def start(self, markets: Iterable[str], *, simulated: bool, use_ai: bool) -> None:
        if self.thread and self.thread.is_alive():
            return

        def _run() -> None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.bot = TradingBot(
                markets=markets,
                simulated=simulated,
                use_ai=use_ai,
                on_update=self.on_update,
            )
            bot_task = self.loop.create_task(self.bot.start())
            bot_task.add_done_callback(lambda _: self.loop.call_soon_threadsafe(self.loop.stop))
            self.loop.run_forever()
            if not bot_task.done():
                bot_task.cancel()
                try:
                    self.loop.run_until_complete(bot_task)
                except Exception:
                    pass
            self.loop.close()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.bot and self.loop:
            self.loop.call_soon_threadsafe(self.bot.stop)
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)


class TradingDashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Upbit 자동매매 대시보드")
        self.root.geometry("1200x800")

        self.status = BotStatus()
        self.runner = BotRunner(self._on_update)
        self.update_queue: "queue.Queue[DecisionUpdate]" = queue.Queue()
        self.latest: Dict[str, DecisionUpdate] = {}
        self.price_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))

        self._build_controls()
        self._build_tables()
        self._build_chart()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_controls(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(frame, text="모니터링 마켓(KRW-BTC,KRW-ETH... 비워두면 전체)").pack(anchor=tk.W)
        self.market_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.market_var).pack(fill=tk.X, pady=2)

        options = ttk.Frame(frame)
        options.pack(fill=tk.X, pady=2)
        self.simulated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="모의주문 모드", variable=self.simulated_var).pack(side=tk.LEFT, padx=5)
        self.ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="OpenAI 의사결정 사용", variable=self.ai_var).pack(side=tk.LEFT, padx=5)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, pady=2)
        self.start_btn = ttk.Button(buttons, text="거래 시작", command=self.start_trading)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(buttons, text="거래 종료", command=self.stop_trading, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(frame, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, pady=2)

    def _build_tables(self) -> None:
        container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left = ttk.Frame(container)
        right = ttk.Frame(container)
        container.add(left, weight=3)
        container.add(right, weight=2)

        columns = ("market", "price", "score", "signal", "reason", "ai")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", height=15)
        headings = {
            "market": "마켓",
            "price": "가격",
            "score": "점수",
            "signal": "신호",
            "reason": "이유",
            "ai": "AI 응답",
        }
        for col, text in headings.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, anchor=tk.W, width=140 if col == "reason" else 100)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.account_frame = ttk.Labelframe(right, text="계좌 현황")
        self.account_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        self.balance_var = tk.StringVar(value="원화 잔고: -")
        self.total_var = tk.StringVar(value="총 평가액: -")
        ttk.Label(self.account_frame, textvariable=self.balance_var).pack(anchor=tk.W, padx=4)
        ttk.Label(self.account_frame, textvariable=self.total_var).pack(anchor=tk.W, padx=4)

        columns = ("market", "balance", "avg", "est")
        self.holding_table = ttk.Treeview(self.account_frame, columns=columns, show="headings", height=6)
        headings = {
            "market": "종목",
            "balance": "보유 수량",
            "avg": "평단가",
            "est": "평가액(원)",
        }
        for col, text in headings.items():
            self.holding_table.heading(col, text=text)
            self.holding_table.column(col, anchor=tk.W, width=120)
        self.holding_table.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        ttk.Label(right, text="최근 이벤트 로그").pack(anchor=tk.W)
        self.log = tk.Text(right, height=10)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _build_chart(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        top = ttk.Frame(frame)
        top.pack(fill=tk.X)
        ttk.Label(top, text="차트 대상 마켓").pack(side=tk.LEFT)
        self.chart_market = tk.StringVar()
        self.market_selector = ttk.Combobox(top, textvariable=self.chart_market, state="readonly")
        self.market_selector.pack(side=tk.LEFT, padx=5)
        self.market_selector.bind("<<ComboboxSelected>>", lambda _: self._refresh_chart())

        fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("가격 추세")
        self.ax.set_xlabel("틱")
        self.ax.set_ylabel("가격")
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_trading(self) -> None:
        markets_text = self.market_var.get().strip()
        markets: List[str]
        if markets_text:
            markets = [m.strip().upper() for m in markets_text.split(",") if m.strip()]
        else:
            markets = fetch_markets(is_fiat=True, fiat_symbol="KRW")

        if not markets:
            messagebox.showerror("마켓 없음", "구독할 마켓을 찾지 못했습니다.")
            return

        self.status = BotStatus(running=True, markets=markets, simulated=self.simulated_var.get(), use_ai=self.ai_var.get())
        self.runner.start(markets, simulated=self.status.simulated, use_ai=self.status.use_ai)
        self.status_var.set(f"실행 중 | 모니터링 {len(markets)}개 마켓")
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self._poll_updates()

    def stop_trading(self) -> None:
        self.runner.stop()
        self.status.running = False
        self.status_var.set("대기 중")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def _on_update(self, update: DecisionUpdate) -> None:
        self.update_queue.put(update)

    def _poll_updates(self) -> None:
        while not self.update_queue.empty():
            update = self.update_queue.get()
            self.latest[update.market] = update
            self.price_history[update.market].append(update.price)
            current_values = list(self.market_selector["values"])
            if update.market not in current_values:
                current_values.append(update.market)
                self.market_selector["values"] = current_values
                if not self.chart_market.get():
                    self.chart_market.set(update.market)
            self._append_log(update)
            if update.account:
                self._update_account(update.account)
        self._refresh_table()
        self._refresh_chart()
        if self.status.running:
            self.root.after(500, self._poll_updates)

    def _append_log(self, update: DecisionUpdate) -> None:
        ts = update.timestamp.strftime("%H:%M:%S")
        order_txt = "실행" if update.executed else "대기"
        ai_txt = update.ai_raw[:80] + ("..." if len(update.ai_raw) > 80 else "")
        line = f"[{ts}] {update.market} {update.signal} 점수 {update.score:.1f} ({order_txt}) | {update.reason} | AI={ai_txt}\n"
        self.log.insert(tk.END, line)
        self.log.see(tk.END)

    def _refresh_table(self) -> None:
        for row in self.tree.get_children():
            self.tree.delete(row)
        sorted_updates = sorted(self.latest.values(), key=lambda u: u.timestamp, reverse=True)[:50]
        for upd in sorted_updates:
            ai_text = "-" if not upd.ai_raw else upd.ai_raw.split("\\n")[0][:60]
            self.tree.insert(
                "",
                tk.END,
                values=(
                    upd.market,
                    f"{upd.price:,.0f}",
                    f"{upd.score:.1f}",
                    upd.signal,
                    upd.reason,
                    ai_text,
                ),
            )

    def _refresh_chart(self) -> None:
        market = self.chart_market.get()
        self.ax.clear()
        self.ax.set_title(f"{market or '마켓 선택'} 가격 추세")
        self.ax.set_xlabel("틱")
        self.ax.set_ylabel("가격")
        if market and market in self.price_history:
            prices = list(self.price_history[market])
            self.ax.plot(prices, color="blue")
        self.canvas.draw_idle()

    def _update_account(self, snapshot: AccountSnapshot) -> None:
        self.balance_var.set(f"원화 잔고: {snapshot.krw_balance:,.0f}원")
        self.total_var.set(f"총 평가액: {snapshot.total_value:,.0f}원")
        for row in self.holding_table.get_children():
            self.holding_table.delete(row)
        for h in snapshot.holdings:
            self.holding_table.insert(
                "",
                tk.END,
                values=(
                    h.market,
                    f"{h.balance:,.6f}",
                    f"{h.avg_buy_price:,.0f}",
                    f"{h.estimated_krw:,.0f}",
                ),
            )

    def _on_close(self) -> None:
        self.stop_trading()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = TradingDashboard()
    app.run()
