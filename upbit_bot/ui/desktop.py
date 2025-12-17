"""PySide6 기반 데스크톱 대시보드.

신호 테이블·계좌·차트·로그를 카드형 그리드로 묶고, 기존
``TradingDashboard``와 동일한 ``on_update`` 이벤트를 어댑터를 통해 수신해
공통 데이터 모델을 재사용한다.
"""
from __future__ import annotations

import asyncio
import queue
import sys
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
    QSortFilterProxyModel,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableView,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
import matplotlib
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from upbit_bot.app import DecisionUpdate, TradingBot
from upbit_bot.data.market_fetcher import fetch_markets
from upbit_bot.data.upbit_adapter import UpbitAdapter
from upbit_bot.indicators import technical
from upbit_bot.storage import SQLiteStateStore

PIN_ROLE = Qt.UserRole + 1
TIMESTAMP_ROLE = Qt.UserRole + 2

_SUPERTREND_CONFIGS: Tuple[Tuple[str, int, float], ...] = (
    ("weak", 7, 2.0),
    ("medium", 10, 3.0),
    ("strong", 14, 4.0),
)

_SUPERTREND_STYLES: Dict[str, Dict[str, Any]] = {
    "weak": {"color": "#f59e0b", "linestyle": "--"},
    "medium": {"color": "#3b82f6", "linestyle": "-.", "alpha": 0.9},
    "strong": {"color": "#ef4444", "linestyle": "-", "linewidth": 1.4},
}


def _format_price_ticks(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs_value >= 1:
        return f"{value:.0f}"
    return f"{value:.2f}"


def _apply_axes_style(ax, chart_palette: Dict[str, str], *, labelsize: int = 9) -> None:
    """차트 축/눈금/그리드를 테마 색상에 맞춰 일괄 적용."""

    axis_color = chart_palette.get("axis", chart_palette.get("text", "#0f172a"))
    spine_color = chart_palette.get("spine", axis_color)
    grid_color = chart_palette.get("grid", spine_color)

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.05)

    ax.tick_params(axis="both", colors=axis_color, labelsize=labelsize)
    ax.grid(True, color=grid_color, alpha=0.45)
    ax.xaxis.label.set_color(chart_palette.get("text", axis_color))
    ax.yaxis.label.set_color(chart_palette.get("text", axis_color))
    ax.title.set_color(chart_palette.get("text", axis_color))


class UpdateAdapter(QObject):
    """TradingBot의 ``on_update``를 받아 Qt 시그널로 전달."""

    update_received = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.queue: "queue.Queue[DecisionUpdate]" = queue.Queue()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.flush)

    def on_update(self, update: DecisionUpdate) -> None:
        self.queue.put(update)

    def start(self) -> None:
        self.timer.start(400)

    def stop(self) -> None:
        self.timer.stop()

    def flush(self) -> None:
        while not self.queue.empty():
            self.update_received.emit(self.queue.get())


class BotRunner:
    """Qt 이벤트루프를 막지 않도록 별도 쓰레드에서 TradingBot 실행."""

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

    def _dispatch(self, func, *args, **kwargs) -> None:
        if self.bot and self.loop:
            self.loop.call_soon_threadsafe(func, *args, **kwargs)

    def set_global_enabled(self, enabled: bool) -> None:
        self._dispatch(self.bot.set_global_enabled, enabled)  # type: ignore[arg-type]

    def set_emergency_stop(self, *, active: bool, close_positions: bool) -> None:
        self._dispatch(self.bot.set_emergency_stop, active=active, close_positions=close_positions)  # type: ignore[arg-type]

    def set_strategy_enabled(self, name: str, enabled: bool) -> None:
        self._dispatch(self.bot.set_strategy_enabled, name, enabled)  # type: ignore[arg-type]


class SignalTableModel(QAbstractTableModel):
    headers = ["★", "마켓", "가격", "점수", "신호", "이유", "AI", "시간"]

    def __init__(self) -> None:
        super().__init__()
        self._latest: Dict[str, DecisionUpdate] = {}
        self._order: List[str] = []
        self._pinned: Set[str] = set()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._order)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        market = self._order[index.row()]
        update = self._latest[market]

        if role in (Qt.DisplayRole, Qt.EditRole):
            col = index.column()
            if col == 0:
                return "★" if market in self._pinned else "☆"
            if col == 1:
                return update.market
            if col == 2:
                return f"{update.price:,.0f}"
            if col == 3:
                return f"{update.score:.1f}"
            if col == 4:
                return update.signal.name
            if col == 5:
                return update.reason
            if col == 6:
                return (update.ai_raw or "-").split("\n")[0][:80]
            if col == 7:
                return update.timestamp.strftime("%H:%M:%S")
        if role == Qt.TextAlignmentRole and index.column() in (2, 3):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == PIN_ROLE:
            return market in self._pinned
        if role == TIMESTAMP_ROLE:
            return update.timestamp.timestamp()
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def flags(self, index: QModelIndex):  # noqa: N802
        base = super().flags(index)
        if index.column() == 0:
            return base | Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
        return base

    def toggle_pin(self, row: int) -> None:
        if row < 0 or row >= len(self._order):
            return
        market = self._order[row]
        if market in self._pinned:
            self._pinned.remove(market)
        else:
            self._pinned.add(market)
        self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [Qt.DisplayRole, PIN_ROLE])

    def upsert(self, update: DecisionUpdate) -> None:
        market = update.market
        is_new = market not in self._latest
        self._latest[market] = update
        if is_new:
            self.beginInsertRows(QModelIndex(), len(self._order), len(self._order))
            self._order.append(market)
            self.endInsertRows()
        self.dataChanged.emit(self.index(0, 0), self.index(len(self._order) - 1, len(self.headers) - 1))


class SignalFilterProxy(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self._query = ""

    def set_query(self, text: str) -> None:
        self._query = text.lower()
        self.invalidate()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        if not self._query:
            return True
        model: SignalTableModel = self.sourceModel()  # type: ignore[assignment]
        market_index = model.index(source_row, 1, source_parent)
        reason_index = model.index(source_row, 5, source_parent)
        ai_index = model.index(source_row, 6, source_parent)
        texts = [
            str(model.data(market_index, Qt.DisplayRole) or "").lower(),
            str(model.data(reason_index, Qt.DisplayRole) or "").lower(),
            str(model.data(ai_index, Qt.DisplayRole) or "").lower(),
        ]
        return any(self._query in t for t in texts)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        model: SignalTableModel = self.sourceModel()  # type: ignore[assignment]
        left_pin = bool(model.data(left, PIN_ROLE))
        right_pin = bool(model.data(right, PIN_ROLE))
        if left_pin != right_pin:
            return right_pin  # pinned(True) should come first
        column = left.column()
        if column == 2:
            left_val = str(model.data(left, Qt.DisplayRole) or "0").replace(",", "")
            right_val = str(model.data(right, Qt.DisplayRole) or "0").replace(",", "")
            return float(left_val or 0.0) < float(right_val or 0.0)
        if column == 3:
            left_val = str(model.data(left, Qt.DisplayRole) or "0").replace(",", "")
            right_val = str(model.data(right, Qt.DisplayRole) or "0").replace(",", "")
            return float(left_val or 0.0) < float(right_val or 0.0)
        if column == 7:
            left_time = float(model.data(left, TIMESTAMP_ROLE) or 0.0)
            right_time = float(model.data(right, TIMESTAMP_ROLE) or 0.0)
            return left_time < right_time
        left_text = str(model.data(left, Qt.DisplayRole) or "")
        right_text = str(model.data(right, Qt.DisplayRole) or "")
        return left_text < right_text


class DesktopDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Upbit PySide6 대시보드")
        self.resize(1400, 900)

        self._configure_matplotlib_fonts()

        self.adapter = UpdateAdapter()
        self.runner = BotRunner(self.adapter.on_update)
        self.adapter.update_received.connect(self._handle_update)
        self.adapter.start()

        self.signal_model = SignalTableModel()
        self.proxy_model = SignalFilterProxy()
        self.proxy_model.setSourceModel(self.signal_model)
        self.price_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
        self.candle_history: Dict[str, List[float]] = defaultdict(list)
        self.latest_account = None
        self.active_markets: List[str] = []
        self.equity_history: Deque[float] = deque(maxlen=400)
        self.heatmap_cache: List[Dict[str, float]] = []
        self.active_orders_model = OrderTableModel()
        self.trade_history_model = TradeHistoryModel()
        self.error_log_model = ErrorLogTableModel()
        self.state_store_reader: Optional[SQLiteStateStore] = None
        self.timeline_seen: set[str] = set()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_state_store)
        self._poll_timer.start(2000)
        self._market_fetch_thread: Optional[threading.Thread] = None
        self._market_fetch_token = 0
        self._market_timeout_timer: Optional[QTimer] = None

        self._init_ui()
        self._apply_dark_theme()
        self._apply_modern_styles()

    # UI 구성
    def _init_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self._build_toolbar())
        layout.addWidget(self._build_banner())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_main_tab(), "메인 대시보드")
        self.tabs.addTab(self._build_positions_tab(), "포지션·주문")
        self.tabs.addTab(self._build_strategy_tab(), "전략 모니터링")
        self.tabs.addTab(self._build_settings_tab(), "설정")
        self.tabs.addTab(self._build_logs_tab(), "로그&알림")
        self.tabs.addTab(self._build_insights_tab(), "인사이트")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(0, 0, 0, 0)

        h.addWidget(QLabel("모니터링 마켓(비워두면 전체):"))
        self.market_input = QLineEdit()
        self.market_input.setPlaceholderText("KRW-BTC,KRW-ETH ...")
        h.addWidget(self.market_input)

        self.simulated_check = QCheckBox("모의주문")
        self.simulated_check.setChecked(True)
        h.addWidget(self.simulated_check)

        self.ai_check = QCheckBox("OpenAI 판단")
        self.ai_check.setChecked(True)
        h.addWidget(self.ai_check)

        self.global_toggle = QToolButton()
        self.global_toggle.setText("전략 전체 ON")
        self.global_toggle.setCheckable(True)
        self.global_toggle.setChecked(True)
        self.global_toggle.toggled.connect(self._handle_global_toggle)
        h.addWidget(self.global_toggle)

        self.all_stop_btn = QPushButton("긴급 ALL STOP")
        self.all_stop_btn.clicked.connect(self._confirm_all_stop)
        h.addWidget(self.all_stop_btn)

        self.start_btn = QPushButton("거래 시작")
        self.start_btn.clicked.connect(self.start_trading)
        h.addWidget(self.start_btn)

        self.stop_btn = QPushButton("종료")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_trading)
        h.addWidget(self.stop_btn)

        h.addStretch(1)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("검색/필터")
        self.search_box.textChanged.connect(self.proxy_model.set_query)
        h.addWidget(self.search_box)

        self.status_label = QLabel("대기 중")
        self.status_label.setMinimumWidth(200)
        h.addWidget(self.status_label)

        self.theme_btn = QToolButton()
        self.theme_btn.setText("라이트")
        self.theme_btn.setCheckable(True)
        self.theme_btn.toggled.connect(self._toggle_theme)
        h.addWidget(self.theme_btn)

        return bar

    def _build_banner(self) -> QWidget:
        self.banner = QLabel("주문/에러 알림이 여기에 표시됩니다.")
        self.banner.setAlignment(Qt.AlignCenter)
        colors = self._theme_palette()["status"]["info"]
        border = self._theme_palette()["banner_border"]
        self.banner.setStyleSheet(
            "QLabel { "
            f"background:{colors['bg']}; color:{colors['text']}; border-radius:6px; padding:6px; "
            f"border: 1px solid {border};"
            "}"
        )
        return self.banner

    def _create_card_frame(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: bold; }")
        return box

    def _build_main_tab(self) -> QWidget:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setSpacing(8)

        # 메트릭 카드
        self.total_asset_label = QLabel("-")
        self.cash_label = QLabel("-")
        self.pnl_label = QLabel("-")
        self.realized_label = QLabel("-")

        metric_labels = [
            ("총자산", self.total_asset_label),
            ("사용 가능 현금", self.cash_label),
            ("평가손익", self.pnl_label),
            ("일일 실현손익", self.realized_label),
        ]
        for idx, (title, label) in enumerate(metric_labels):
            box = self._create_card_frame(title)
            v = QVBoxLayout(box)
            label.setStyleSheet("font-size: 18px; font-weight: bold;")
            v.addWidget(label)
            grid.addWidget(box, 0, idx, 1, 1)

        # 리스크/연결 상태 배지
        status_box = self._create_card_frame("리스크 & 연결 상태")
        status_layout = QVBoxLayout(status_box)
        self.loss_limit_badge = QLabel("일일 손실 한도: -")
        self.position_limit_badge = QLabel("포지션 비중: -")
        self.api_status_badge = QLabel("API 상태: -")
        self.ws_status_badge = QLabel("WebSocket: -")
        for badge in [self.loss_limit_badge, self.position_limit_badge, self.api_status_badge, self.ws_status_badge]:
            badge.setFrameShape(QFrame.Panel)
            badge.setFrameShadow(QFrame.Raised)
            badge.setStyleSheet("padding:6px; border-radius:6px;")
            status_layout.addWidget(badge)
        grid.addWidget(status_box, 0, len(metric_labels), 1, 1)

        # 실시간 에쿼티 곡선
        chart_box = self._create_card_frame("실시간 에쿼티 곡선")
        chart_layout = QVBoxLayout(chart_box)
        fig = Figure(figsize=(8, 3), dpi=100)
        self.equity_ax = fig.add_subplot(111)
        self.equity_canvas = FigureCanvasQTAgg(fig)
        self.equity_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.equity_canvas)
        grid.addWidget(chart_box, 1, 0, 1, len(metric_labels) + 1)

        # 반응형 배치 보정
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(4, 2)

        return tab

    def _build_positions_tab(self) -> QWidget:
        tab = QWidget()
        splitter = QSplitter(Qt.Horizontal, tab)

        holdings_box = self._create_card_frame("보유 코인 상세")
        holdings_layout = QVBoxLayout(holdings_box)
        self.favorite_box = QComboBox()
        self.favorite_box.setPlaceholderText("즐겨찾기 마켓")
        holdings_layout.addWidget(self.favorite_box)
        self.holding_table = QTableView()
        self.holding_model = HoldingTableModel()
        self.holding_view_proxy = QSortFilterProxyModel()
        self.holding_view_proxy.setSourceModel(self.holding_model)
        self.holding_view_proxy.setSortRole(TIMESTAMP_ROLE)
        self.holding_table.setModel(self.holding_view_proxy)
        self.holding_table.setSortingEnabled(True)
        self.holding_table.horizontalHeader().setStretchLastSection(True)
        self.holding_table.setAlternatingRowColors(True)
        holdings_layout.addWidget(self.holding_table)

        orders_box = self._create_card_frame("진행 중인 주문")
        orders_layout = QVBoxLayout(orders_box)
        self.order_table = QTableView()
        self.order_table.setModel(self.active_orders_model)
        self.order_table.horizontalHeader().setStretchLastSection(True)
        self.order_table.setAlternatingRowColors(True)
        orders_layout.addWidget(self.order_table)

        trades_box = self._create_card_frame("체결 히스토리")
        trades_layout = QVBoxLayout(trades_box)
        self.trade_table = QTableView()
        self.trade_table.setModel(self.trade_history_model)
        self.trade_table.horizontalHeader().setStretchLastSection(True)
        self.trade_table.setAlternatingRowColors(True)
        trades_layout.addWidget(self.trade_table)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addWidget(orders_box)
        right_layout.addWidget(trades_box)

        splitter.addWidget(holdings_box)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout(tab)
        outer.addWidget(splitter)
        return tab

    def _build_strategy_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("전략별 ON/OFF"))
        self.strategy_switches: Dict[str, QCheckBox] = {}
        for display, key in [("Supertrend", "supertrend")]:
            chk = QCheckBox(display)
            chk.setChecked(True)
            chk.toggled.connect(lambda state, n=key: self._toggle_strategy(n, state))
            controls.addWidget(chk)
            self.strategy_switches[key] = chk
        controls.addStretch(1)
        layout.addLayout(controls)

        viz_splitter = QSplitter(Qt.Horizontal)

        heatmap_box = self._create_card_frame("코인별 스코어 Heatmap")
        heatmap_layout = QVBoxLayout(heatmap_box)
        chart_palette = self._theme_palette().get("chart", {})
        heatmap_bg = chart_palette.get("bg", self._theme_palette().get("card"))
        heatmap_fig = Figure(figsize=(4, 3), dpi=100)
        heatmap_fig.patch.set_facecolor(heatmap_bg)
        self.heatmap_ax = heatmap_fig.add_subplot(111, facecolor=heatmap_bg)
        self.heatmap_canvas = FigureCanvasQTAgg(heatmap_fig)
        self.heatmap_canvas.setStyleSheet(f"background-color: {heatmap_bg};")
        heatmap_layout.addWidget(self.heatmap_canvas)

        chart_box = self._create_card_frame("캔들 + 지표 + 매수/매도 마커")
        chart_layout = QVBoxLayout(chart_box)
        top = QHBoxLayout()
        top.addWidget(QLabel("대상 마켓"))
        self.chart_selector = QComboBox()
        self.chart_selector.currentTextChanged.connect(self._refresh_chart)
        top.addWidget(self.chart_selector)
        self.refresh_candles_btn = QPushButton("캔들 새로고침")
        self.refresh_candles_btn.clicked.connect(self._refresh_chart)
        top.addWidget(self.refresh_candles_btn)
        top.addStretch(1)
        chart_layout.addLayout(top)

        fig = Figure(figsize=(6, 3), dpi=100)
        chart_palette = self._theme_palette().get("chart", {})
        chart_bg = chart_palette.get("bg", self._theme_palette().get("card"))
        fig.patch.set_facecolor(chart_bg)
        self.ax = fig.add_subplot(111, facecolor=chart_bg)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setStyleSheet(f"background-color: {chart_bg};")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.canvas)

        self.signal_view = QTableView()
        self.signal_view.setModel(self.proxy_model)
        self.signal_view.setSortingEnabled(True)
        self.signal_view.sortByColumn(7, Qt.DescendingOrder)
        self.signal_view.clicked.connect(self._handle_table_click)
        self.signal_view.horizontalHeader().setStretchLastSection(True)

        signal_box = self._create_card_frame("신호 테이블")
        signal_layout = QVBoxLayout(signal_box)
        signal_layout.addWidget(self.signal_view)

        viz_splitter.addWidget(heatmap_box)
        viz_splitter.addWidget(chart_box)
        viz_splitter.addWidget(signal_box)
        viz_splitter.setStretchFactor(0, 1)
        viz_splitter.setStretchFactor(1, 2)
        viz_splitter.setStretchFactor(2, 2)

        layout.addWidget(viz_splitter)
        return tab

    def _build_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)
        layout.setSpacing(10)

        # 리스크 파라미터
        layout.addWidget(QLabel("1회 위험 %"), 0, 0)
        self.risk_pct_input = QLineEdit("1.0")
        layout.addWidget(self.risk_pct_input, 0, 1)

        layout.addWidget(QLabel("일일 손실 한도 %"), 1, 0)
        self.daily_loss_input = QLineEdit("5.0")
        layout.addWidget(self.daily_loss_input, 1, 1)

        layout.addWidget(QLabel("코인당 최대 비중 %"), 2, 0)
        self.max_weight_input = QLineEdit("25.0")
        layout.addWidget(self.max_weight_input, 2, 1)

        # 전략 파라미터
        layout.addWidget(QLabel("전략 파라미터"), 0, 2)
        self.strategy_param_input = QLineEdit("예: 기간=20, 가중치=0.5")
        layout.addWidget(self.strategy_param_input, 0, 3)

        layout.addWidget(QLabel("시스템 로그 레벨"), 1, 2)
        self.log_level_input = QComboBox()
        self.log_level_input.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        layout.addWidget(self.log_level_input, 1, 3)

        layout.addWidget(QLabel("알림 채널"), 2, 2)
        self.alert_channel_input = QComboBox()
        self.alert_channel_input.addItems(["Slack", "Telegram", "Email"])
        layout.addWidget(self.alert_channel_input, 2, 3)

        self.preview_label = QLabel("적용 전 미리보기 / 재시작 필요 여부를 확인하세요.")
        layout.addWidget(self.preview_label, 3, 0, 1, 4)
        return tab

    def _build_logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        timeline_box = self._create_card_frame("이벤트 타임라인")
        timeline_layout = QVBoxLayout(timeline_box)
        self.timeline_list = QListWidget()
        timeline_layout.addWidget(self.timeline_list)

        error_box = self._create_card_frame("에러 로그")
        error_layout = QVBoxLayout(error_box)
        self.error_table = QTableView()
        self.error_table.setModel(self.error_log_model)
        self.error_table.horizontalHeader().setStretchLastSection(True)
        error_layout.addWidget(self.error_table)

        text_box = self._create_card_frame("상세 로그")
        text_layout = QVBoxLayout(text_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        text_layout.addWidget(self.log_view)

        layout.addWidget(timeline_box)
        layout.addWidget(error_box)
        layout.addWidget(text_box)
        return tab

    def _build_insights_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        factor_box = self._create_card_frame("Factor Snapshot")
        factor_layout = QFormLayout(factor_box)
        self.factor_labels: Dict[str, QLabel] = {}
        for key in ["composite", "trend", "momentum", "volatility", "volume", "regime"]:
            lbl = QLabel("-")
            lbl.setStyleSheet("font-size:16px; font-weight:bold;")
            factor_layout.addRow(key.capitalize(), lbl)
            self.factor_labels[key] = lbl

        heatmap_box = self._create_card_frame("Heatmap Top Picks")
        heatmap_layout = QVBoxLayout(heatmap_box)
        self.heatmap_list = QListWidget()
        heatmap_layout.addWidget(self.heatmap_list)

        layout.addWidget(factor_box, 1)
        layout.addWidget(heatmap_box, 2)
        return tab

    # 이벤트 처리
    def start_trading(self) -> None:
        if self._market_fetch_thread and self._market_fetch_thread.is_alive():
            self._show_banner("이미 마켓 조회 중입니다. 잠시만 기다려주세요.", success=False, severity="warn")
            return

        text = self.market_input.text().strip()
        if text:
            markets = [m.strip().upper() for m in text.split(",") if m.strip()]
            self._start_with_markets(markets)
            return

        self.start_btn.setEnabled(False)
        self.status_label.setText("마켓 조회 중...")

        self._market_fetch_token += 1
        token = self._market_fetch_token

        if self._market_timeout_timer:
            self._market_timeout_timer.stop()
        self._market_timeout_timer = QTimer(self)
        self._market_timeout_timer.setSingleShot(True)
        self._market_timeout_timer.timeout.connect(lambda: self._on_market_fetch_timeout(token))
        # 마켓 조회 시 거래대금 순위를 계산하기 위해 여러 REST 호출이 필요하다.
        # 네트워크가 살짝 지연되면 12초 제한을 넘기는 경우가 있어 여유를 둔다.
        self._market_timeout_timer.start(20000)

        def _fetch_markets() -> None:
            try:
                markets_result = fetch_markets(
                    is_fiat=True,
                    fiat_symbol="KRW",
                    top_by_volume=5,
                    adapter=UpbitAdapter(timeout=4, max_retries=1, backoff=0.3),
                )
            except Exception as exc:  # pragma: no cover - 네트워크 예외 방어
                QTimer.singleShot(0, lambda: self._handle_market_fetch_failure(exc, token))
                return
            QTimer.singleShot(0, lambda: self._finish_market_fetch(markets_result, token))

        self._market_fetch_thread = threading.Thread(target=_fetch_markets, daemon=True)
        self._market_fetch_thread.start()

    def _finish_market_fetch(self, markets: List[str], token: int) -> None:
        if token != self._market_fetch_token:
            return
        self._clear_market_fetch_timer()
        self._market_fetch_thread = None
        self._start_with_markets(markets)

    def _start_with_markets(self, markets: List[str]) -> None:
        if not markets:
            QMessageBox.warning(self, "마켓 없음", "구독할 마켓을 찾지 못했습니다.")
            self.start_btn.setEnabled(True)
            self.status_label.setText("대기 중")
            return

        self.runner.start(markets, simulated=self.simulated_check.isChecked(), use_ai=self.ai_check.isChecked())
        self.active_markets = markets
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._show_banner(f"{len(markets)}개 마켓 구독 시작", success=True, severity="success")
        self.status_label.setText(f"실행 중 | 모니터링 {len(markets)}개")
        self._add_timeline_event("거래 시작", severity="success")

    def _on_market_fetch_timeout(self, token: int) -> None:
        if token != self._market_fetch_token:
            return
        self._market_fetch_thread = None
        self._handle_market_fetch_failure(TimeoutError("마켓 조회가 제한 시간 내 완료되지 않았습니다."), token)

    def _clear_market_fetch_timer(self) -> None:
        if self._market_timeout_timer:
            self._market_timeout_timer.stop()
            self._market_timeout_timer = None

    def _handle_market_fetch_failure(self, exc: Exception, token: Optional[int] = None) -> None:
        if token is not None and token != self._market_fetch_token:
            return
        self._clear_market_fetch_timer()
        self._market_fetch_thread = None
        self._show_banner(f"마켓 조회 실패: {exc}", success=False, severity="error")
        self.start_btn.setEnabled(True)
        self.status_label.setText("대기 중")

    def stop_trading(self) -> None:
        self.runner.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._show_banner("거래 종료", success=True, severity="warn")
        self.status_label.setText("대기 중")
        self.active_markets = []
        self._add_timeline_event("거래 종료", severity="warn")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.adapter.stop()
        self.runner.stop()
        return super().closeEvent(event)

    def _handle_table_click(self, index: QModelIndex) -> None:
        if index.column() != 0:
            return
        source = self.proxy_model.mapToSource(index)
        self.signal_model.toggle_pin(source.row())
        market = self.signal_model._order[source.row()]
        if market in self.signal_model._pinned and self.favorite_box.findText(market) == -1:
            self.favorite_box.addItem(market)
        elif market not in self.signal_model._pinned:
            idx = self.favorite_box.findText(market)
            if idx != -1:
                self.favorite_box.removeItem(idx)

    def _handle_update(self, update: DecisionUpdate) -> None:
        self.signal_model.upsert(update)
        self.proxy_model.invalidate()
        self.price_history[update.market].append(update.price)
        if self.chart_selector.findText(update.market) == -1:
            self.chart_selector.addItem(update.market)
        self._refresh_chart()
        self._refresh_heatmap()

        if update.account:
            self.latest_account = update.account
            self._update_account(update.account)

        self._append_log(update)

        if update.executed:
            self._show_banner(
                f"{update.market} {update.signal.name} 주문 처리",
                success=True,
                severity="success",
            )
            self.trade_history_model.add_trade(update)
        if update.order_result and update.order_result.error:
            self._show_banner(update.order_result.error, success=False, severity="error")
            self.error_log_model.add_error(update.order_result.error, severity="error")
        if update.order_result:
            self.active_orders_model.add_order(update.order_result)
            self._add_timeline_event(
                f"{update.market} {update.order_result.side.upper()} 수량 {update.order_result.volume:.4f}",
                severity="info" if update.order_result.success else "warn",
            )

    def _append_log(self, update: DecisionUpdate) -> None:
        if update.suppress_log:
            return
        ts = update.timestamp.strftime("%H:%M:%S")
        ai_txt = (update.ai_raw or "-").split("\n")[0]
        ai_short = ai_txt[:80] + ("..." if len(ai_txt) > 80 else "")
        status = "실행" if update.executed else "대기"
        line = f"[{ts}] {update.market} {update.signal.name} 점수 {update.score:.1f} ({status}) | {update.reason} | AI={ai_short}\n"
        self.log_view.append(line)
        severity = "info"
        if update.order_result and update.order_result.error:
            severity = "error"
        elif update.executed:
            severity = "success"
        self._add_timeline_event(line.strip(), severity=severity)

    def _update_account(self, snapshot) -> None:
        self.cash_label.setText(f"{snapshot.krw_balance:,.0f}원")
        self.total_asset_label.setText(f"{snapshot.total_value:,.0f}원")
        self.pnl_label.setText(f"{snapshot.profit:,.0f}원 ({snapshot.profit_pct:+.2f}%)")
        self.realized_label.setText(f"일일 실현손익 추정: {snapshot.profit:,.0f}원")
        self.holding_model.update(snapshot.holdings)
        self.holding_view_proxy.invalidate()
        self.equity_history.append(snapshot.total_value)
        self._refresh_equity_curve()
        self._update_status_badges(snapshot)

    def _refresh_chart(self) -> None:
        market = self.chart_selector.currentText()
        self.ax.clear()
        palette = self._theme_palette()
        chart_palette = palette.get(
            "chart",
            {
                "bg": palette.get("card", "white"),
                "grid": palette.get("border", "#e2e8f0"),
                "spine": palette.get("border", "#e2e8f0"),
                "text": palette.get("text", "#0f172a"),
                "legend_face": palette.get("card", "white"),
                "legend_edge": palette.get("border", "#e2e8f0"),
                "bull": "#22c55e",
                "bear": "#ef4444",
                "ema_fast": palette.get("accent_alt", "#22c55e"),
                "ema_slow": "#a855f7",
                "ema_long": "#f97316",
                "fill": palette.get("accent", "#2563eb"),
                "trade_buy": "#f59e0b",
                "trade_sell": "#10b981",
                "heatmap_cmap": "magma",
                "equity_line": "#22c55e",
                "supertrend": {},
            },
        )
        bg_color = chart_palette.get("bg", palette.get("card"))
        self.ax.figure.set_facecolor(bg_color)
        self.ax.figure.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(chart_palette.get("bg"))
        self.canvas.setStyleSheet(f"background-color: {bg_color};")
        self.ax.set_title(market or "Select Market", color=chart_palette.get("text"))
        legend_handles: List[Line2D] = []
        if market:
            frame = self.runner.bot.ohlcv_service.get_frame(market, "5m") if self.runner.bot else None
            if frame is not None and not frame.empty:
                lookback = min(len(frame), 150)
                closes = frame["close"].tail(lookback)
                highs = frame["high"].tail(lookback)
                lows = frame["low"].tail(lookback)
                opens = frame["open"].tail(lookback)
                x = np.arange(len(closes))
                for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
                    color = chart_palette.get("bull") if c >= o else chart_palette.get("bear")
                    self.ax.vlines(i, l, h, color=color, alpha=0.65)
                    self.ax.vlines(i, min(o, c), max(o, c), color=color, linewidth=6, alpha=0.9)

                legend_handles.extend(
                    [
                        Line2D([], [], color=chart_palette.get("bull"), linewidth=4, label="상승 캔들"),
                        Line2D([], [], color=chart_palette.get("bear"), linewidth=4, label="하락 캔들"),
                    ]
                )

                ema200_line: Optional[Line2D] = None
                if len(frame) >= 200:
                    ema200_series = technical.ema(frame["close"], 200).iloc[-len(closes) :]
                    ema200_line = self.ax.plot(
                        x[-len(ema200_series) :],
                        ema200_series,
                        color=chart_palette.get("ema_long", "#f97316"),
                        label="EMA200",
                        linewidth=1.8,
                        alpha=0.8,
                    )[0]
                    legend_handles.append(ema200_line)

                if len(closes) >= 9:
                    ema9 = closes.ewm(span=9).mean()
                    ema21 = closes.ewm(span=21).mean()
                    ema9_line = self.ax.plot(
                        x,
                        ema9,
                        color=chart_palette.get("ema_fast", "#22d3ee"),
                        label="EMA9",
                        linewidth=1.0,
                        alpha=0.85,
                    )[0]
                    ema21_line = self.ax.plot(
                        x,
                        ema21,
                        color=chart_palette.get("ema_slow", "#a855f7"),
                        label="EMA21",
                        linewidth=1.0,
                        alpha=0.85,
                    )[0]
                    legend_handles.extend([ema9_line, ema21_line])

                supertrend_handles: List[Line2D] = []
                for name, period, mult in _SUPERTREND_CONFIGS:
                    if len(frame) < max(period, 5):
                        continue
                    st_df = technical.supertrend(frame["close"], period=period, multiplier=mult)
                    series = st_df["supertrend"].iloc[-len(closes) :]
                    if series.empty:
                        continue
                    styles = dict(_SUPERTREND_STYLES.get(name, {}))
                    st_colors = chart_palette.get("supertrend", {})
                    if name in st_colors:
                        styles["color"] = st_colors[name]
                    styles.setdefault("alpha", 0.85)
                    linewidth = styles.pop("linewidth", 1.2)
                    st_line = self.ax.plot(
                        x[-len(series) :],
                        series,
                        label=f"Supertrend {name}",
                        linewidth=linewidth,
                        **styles,
                    )[0]
                    supertrend_handles.append(st_line)
                legend_handles.extend(supertrend_handles)

                self.ax.fill_between(x, closes, color=chart_palette.get("fill"), alpha=0.08)
                # trade markers
                trades = (self.state_store_reader.load_recent_trades(limit=50) if self.state_store_reader else [])
                has_buy_trade = False
                has_sell_trade = False
                for t in trades:
                    if t.get("market") == market:
                        idx = len(closes) - 1
                        marker_y = t.get("price", closes.iloc[-1])
                        is_sell = t.get("side") == "ask"
                        color = chart_palette.get("trade_sell") if is_sell else chart_palette.get("trade_buy")
                        marker = "^" if is_sell else "v"
                        self.ax.scatter(idx, marker_y, color=color, marker=marker, zorder=6, alpha=0.9)
                        has_sell_trade = has_sell_trade or is_sell
                        has_buy_trade = has_buy_trade or not is_sell

                if has_buy_trade:
                    legend_handles.append(
                        Line2D([], [], color=chart_palette.get("trade_buy"), marker="v", linestyle="None", label="매수 체결")
                    )
                if has_sell_trade:
                    legend_handles.append(
                        Line2D([], [], color=chart_palette.get("trade_sell"), marker="^", linestyle="None", label="매도 체결")
                    )

                max_high = float(highs.max())
                min_low = float(lows.min())
                pad = max((max_high - min_low) * 0.08, 1)
                self.ax.set_ylim(min_low - pad, max_high + pad)
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: _format_price_ticks(val)))
                self.ax.margins(y=0.05)
        self.ax.set_xlabel("Ticks")
        self.ax.set_ylabel("Price")
        if legend_handles:
            legend = self.ax.legend(handles=legend_handles, loc="upper left")
            legend.get_frame().set_facecolor(chart_palette.get("legend_face"))
            legend.get_frame().set_edgecolor(chart_palette.get("legend_edge"))
            for text in legend.get_texts():
                text.set_color(chart_palette.get("text"))
        # 전략 모니터링 차트도 다크 테마에서 축과 그리드가 묻히지 않도록 테마 색상으로 통일
        _apply_axes_style(self.ax, chart_palette, labelsize=9)
        self.canvas.draw_idle()

    def _refresh_heatmap(self) -> None:
        entries = self.heatmap_cache
        if not entries:
            return
        palette = self._theme_palette()
        chart_palette = palette.get("chart", {})
        bg_color = chart_palette.get("bg", palette.get("card"))
        self.heatmap_ax.figure.set_facecolor(bg_color)
        self.heatmap_ax.figure.patch.set_facecolor(bg_color)
        self.heatmap_canvas.setStyleSheet(f"background-color: {bg_color};")
        markets = sorted(entries, key=lambda e: e.get("composite", 0), reverse=True)[:16]
        scores = [m.get("composite", 0) * 100 for m in markets]
        size = int(len(scores) ** 0.5) or 1
        while size * size < len(scores):
            size += 1
        grid_scores = scores + [0.0] * (size * size - len(scores))
        matrix = np.array(grid_scores, dtype=float).reshape(size, size)
        self.heatmap_ax.clear()
        self.heatmap_ax.set_facecolor(bg_color)
        cmap_name = chart_palette.get("heatmap_cmap", "magma")
        cmap = cm.get_cmap(cmap_name)
        vmin, vmax = float(matrix.min()), float(matrix.max())
        if vmax == vmin:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        _ = self.heatmap_ax.imshow(matrix, cmap=cmap, norm=norm)
        self.heatmap_ax.set_xticks([])
        self.heatmap_ax.set_yticks([])
        self.heatmap_ax.set_title("Score Heatmap", color=chart_palette.get("text", palette.get("text")))

        # 다크 테마에서도 축/테두리가 흐릿하게 보이지 않도록 팔레트 기반 스타일 적용
        _apply_axes_style(self.heatmap_ax, chart_palette, labelsize=8)

        def _contrast_color(score_value: float) -> str:
            rgba = cmap(norm(score_value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            if luminance > 0.6:
                return "#0f172a"
            return "#f8fafc"

        for idx, score in enumerate(scores):
            y, x = divmod(idx, size)
            label = f"{markets[idx].get('market','')}\n{score:.1f}"
            self.heatmap_ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                color=_contrast_color(score),
                fontweight="bold",
            )
        self.heatmap_canvas.draw_idle()

    def _refresh_heatmap_list(self, entries: List[Dict[str, float]]) -> None:
        self.heatmap_list.clear()
        top = sorted(entries, key=lambda e: e.get("composite", 0), reverse=True)[:20]
        for item in top:
            text = f"{item.get('market','')} | C {item.get('composite',0)*100:.1f} / T {item.get('trend',0)*100:.1f} / M {item.get('momentum',0)*100:.1f}"
            self.heatmap_list.addItem(text)

    def _refresh_equity_curve(self) -> None:
        self.equity_ax.clear()
        palette = self._theme_palette()
        chart_palette = palette.get("chart", {})
        self.equity_ax.figure.set_facecolor(chart_palette.get("bg", palette.get("card")))
        self.equity_ax.set_facecolor(chart_palette.get("bg", palette.get("card")))
        if self.equity_history:
            self.equity_ax.plot(
                list(self.equity_history),
                color=chart_palette.get("equity_line", palette.get("accent", "tab:green")),
                alpha=0.9,
                linewidth=1.6,
            )
        # 축/스파인/눈금 색상을 테마에 맞춰 덮어써 다크 모드에서도 대비를 확보
        self.equity_ax.set_xlabel("Update")
        self.equity_ax.set_ylabel("Total Asset")
        self.equity_ax.grid(True, color=chart_palette.get("grid", palette.get("border")), alpha=0.25)
        _apply_axes_style(self.equity_ax, chart_palette, labelsize=9)
        self.equity_canvas.draw_idle()

    def _poll_state_store(self) -> None:
        if not self.state_store_reader:
            db_path = getattr(getattr(self.runner.bot, "state_store", None), "db_path", Path("./.state/trading.db"))
            self.state_store_reader = SQLiteStateStore(db_path=db_path)
            self.state_store_reader.ensure_schema()
        try:
            curve = self.state_store_reader.load_equity_curve(limit=200)
            if curve:
                self.equity_history.clear()
                self.equity_history.extend([c["total_value"] for c in curve])
                self._refresh_equity_curve()
            state = self.state_store_reader.load_strategy_state() or {}
            heatmap = state.get("heatmap")
            if heatmap:
                self.heatmap_cache = heatmap
                self._refresh_heatmap()
            # update factor tiles from first heatmap entry
            if heatmap and self.factor_labels:
                top = heatmap[0]
                for key, lbl in self.factor_labels.items():
                    val = top.get(key)
                    if val is None:
                        lbl.setText("-")
                    else:
                        lbl.setText(f"{val:.2f}" if isinstance(val, float) else str(val))
                self._refresh_heatmap_list(heatmap)
            # populate timeline with orders and risk events
            orders = self.state_store_reader.load_recent_orders(limit=20)
            risks = self.state_store_reader.load_risk_events(limit=10)
            for o in orders:
                key = f"order-{o.get('created_at')}-{o.get('uuid')}"
                if key in self.timeline_seen:
                    continue
                self.timeline_seen.add(key)
                text = f"ORDER {o.get('market')} {o.get('side')} {o.get('price')} x {o.get('volume')} [{o.get('status')}]"
                severity = "success" if o.get("status") == "done" else "warn"
                self._add_timeline_event(text, severity=severity)
            for r in risks:
                key = f"risk-{r.get('created_at')}-{r.get('market')}"
                if key in self.timeline_seen:
                    continue
                self.timeline_seen.add(key)
                text = f"RISK {r.get('market')}: {r.get('reason')}"
                self._add_timeline_event(text, severity="error")
        except Exception as exc:  # pragma: no cover - UI resilience
            self._add_timeline_event(f"상태 스토어 읽기 오류: {exc}", severity="warn")
        finally:
            # 주기 호출에서 예외 후에도 타이머 루프가 멈추지 않도록 보장
            pass

    def _update_status_badges(self, snapshot) -> None:
        bot = self.runner.bot
        loss_limit = bot.daily_loss_limit_pct if bot else 0.0
        max_pos = bot.max_position_pct if bot else 0.0
        max_portfolio = bot.max_portfolio_pct if bot else 0.0
        self.loss_limit_badge.setText(f"일일 손실 한도: {loss_limit:.2f}%")
        hit_limit = loss_limit > 0 and snapshot.profit_pct <= -loss_limit
        self._set_badge_style(self.loss_limit_badge, "error" if hit_limit else "info")
        self.position_limit_badge.setText(
            f"포지션/포트폴리오 비중: {max_pos:.1f}% / {max_portfolio:.1f}%"
        )
        self._set_badge_style(self.position_limit_badge, "info")
        api_state = "실거래" if bot and not bot.simulated else "모의" if bot else "준비중"
        self.api_status_badge.setText(f"API 상태: {api_state}")
        self._set_badge_style(self.api_status_badge, "success" if api_state == "실거래" else "info")
        ws_state = "대기" if not self.active_markets else "연결 유지"
        self.ws_status_badge.setText(f"WebSocket: {ws_state}")
        ws_severity = "success" if ws_state == "연결 유지" else "info"
        self._set_badge_style(self.ws_status_badge, ws_severity)

    def _configure_matplotlib_fonts(self) -> None:
        """한글 글꼴을 우선 적용해 그래프 깨짐과 경고를 방지."""

        candidates = [
            "Malgun Gothic",
            "AppleGothic",
            "NanumGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
        ]

        for name in candidates:
            try:
                font_manager.findfont(name, fallback_to_default=False)
            except ValueError:
                continue
            matplotlib.rcParams["font.family"] = name
            break
        else:
            matplotlib.rcParams["font.family"] = "DejaVu Sans"

        matplotlib.rcParams["axes.unicode_minus"] = False

    def _set_badge_style(self, label: QLabel, severity: str) -> None:
        palette = self._theme_palette()
        colors = palette["status"].get(severity, palette["status"]["info"])
        label.setStyleSheet(
            "padding:6px; border-radius:6px; "
            f"background:{colors['bg']}; color:{colors['text']}; "
            f"border: 1px solid {palette['border']};"
        )

    def _add_timeline_event(self, text: str, *, severity: str = "info") -> None:
        item = QListWidgetItem(text)
        palette = self._theme_palette()
        color_map = palette["status"]
        colors = color_map.get(severity, color_map["info"])
        item.setBackground(QColor(colors["bg"]))
        item.setForeground(QColor(colors["text"]))
        self.timeline_list.insertItem(0, item)
        if self.timeline_list.count() > 80:
            self.timeline_list.takeItem(self.timeline_list.count() - 1)

    def _handle_global_toggle(self, checked: bool) -> None:
        self.global_toggle.setText("전략 전체 ON" if checked else "전략 전체 OFF")
        self.runner.set_global_enabled(checked)
        status = "전략 전체 활성" if checked else "전략 전체 비활성"
        self.status_label.setText(status)
        self._add_timeline_event(status, severity="warn" if not checked else "info")

    def _confirm_all_stop(self) -> None:
        first = QMessageBox.question(
            self,
            "긴급 ALL STOP",
            "신규 진입을 모두 중단하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if first != QMessageBox.Yes:
            return
        second = QMessageBox()
        second.setIcon(QMessageBox.Warning)
        second.setText("기존 포지션을 어떻게 처리할까요?")
        second.setInformativeText("Yes=즉시 청산, No=유지")
        second.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        choice = second.exec()
        if choice == QMessageBox.Cancel:
            return
        close_positions = choice == QMessageBox.Yes
        self.runner.set_emergency_stop(active=True, close_positions=close_positions)
        msg = (
            "ALL STOP: 기존 포지션 청산 및 신규 진입 차단"
            if close_positions
            else "ALL STOP: 신규 진입 차단, 기존 포지션 유지"
        )
        self._show_banner(msg, success=False)
        self.status_label.setText(msg)
        self._add_timeline_event(msg, severity="error" if close_positions else "warn")

    def _toggle_strategy(self, name: str, enabled: bool) -> None:
        self.runner.set_strategy_enabled(name, enabled)
        msg = f"{name} 전략 {'ON' if enabled else 'OFF'}"
        self._add_timeline_event(msg, severity="info" if enabled else "warn")

    def _show_banner(self, text: str, *, success: bool, severity: Optional[str] = None) -> None:
        palette = self._theme_palette()
        status_colors = palette["status"]
        color_key = severity or ("success" if success else "warn")
        colors = status_colors.get(color_key, status_colors["info"])
        self.banner.setText(text)
        self.banner.setStyleSheet(
            "QLabel { "
            f"background:{colors['bg']}; color:{colors['text']}; border-radius:6px; padding:6px; "
            f"border: 1px solid {palette['banner_border']};"
            "}"
        )

    def _toggle_theme(self, checked: bool) -> None:
        if checked:
            self._apply_dark_theme()
            self.theme_btn.setText("다크")
        else:
            self._apply_light_theme()
            self.theme_btn.setText("라이트")
        # 다크/라이트 전환 시 차트 팔레트를 즉시 갱신하여 시각 요소가 밝아지지 않도록 처리
        self._refresh_chart()
        self._refresh_heatmap()
        self._refresh_equity_curve()

    def _apply_dark_theme(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        self.is_dark = True
        palette_config = self._theme_palette()
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(palette_config["bg"]))
        palette.setColor(QPalette.WindowText, QColor(palette_config["text"]))
        palette.setColor(QPalette.Base, QColor(palette_config["card"]))
        palette.setColor(QPalette.AlternateBase, QColor(palette_config["table_alt"]))
        palette.setColor(QPalette.ToolTipBase, QColor(palette_config["tooltip_bg"]))
        palette.setColor(QPalette.ToolTipText, QColor(palette_config["text"]))
        palette.setColor(QPalette.Text, QColor(palette_config["text"]))
        palette.setColor(QPalette.Button, QColor(palette_config["card"]))
        palette.setColor(QPalette.ButtonText, QColor(palette_config["text"]))
        palette.setColor(QPalette.BrightText, QColor(palette_config["accent_alt"]))
        palette.setColor(QPalette.Highlight, QColor(palette_config["selection_bg"]))
        palette.setColor(QPalette.HighlightedText, QColor(palette_config["selection_text"]))
        app.setPalette(palette)
        self._apply_modern_styles()

    def _apply_light_theme(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        self.is_dark = False
        palette_config = self._theme_palette(light=True)
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(palette_config["bg"]))
        palette.setColor(QPalette.WindowText, QColor(palette_config["text"]))
        palette.setColor(QPalette.Base, QColor(palette_config["card"]))
        palette.setColor(QPalette.AlternateBase, QColor(palette_config["table_alt"]))
        palette.setColor(QPalette.ToolTipBase, QColor(palette_config["tooltip_bg"]))
        palette.setColor(QPalette.ToolTipText, QColor(palette_config["text"]))
        palette.setColor(QPalette.Text, QColor(palette_config["text"]))
        palette.setColor(QPalette.Button, QColor(palette_config["card"]))
        palette.setColor(QPalette.ButtonText, QColor(palette_config["text"]))
        palette.setColor(QPalette.BrightText, QColor(palette_config["accent_alt"]))
        palette.setColor(QPalette.Highlight, QColor(palette_config["selection_bg"]))
        palette.setColor(QPalette.HighlightedText, QColor(palette_config["selection_text"]))
        app.setPalette(palette)
        self._apply_modern_styles(light=True)

    def _apply_modern_styles(self, light: bool = False) -> None:
        palette = self._theme_palette(light)
        accent = palette["accent"]
        bg = palette["bg"]
        card = palette["card"]
        border = palette["border"]
        text = palette["text"]
        tab = palette["tab_bg"]
        input_bg = palette["input_bg"]
        muted = palette["muted"]
        header_bg = palette["header_bg"]
        header_text = palette["header_text"]
        self.setStyleSheet(
            f"""
            QWidget {{ background-color: {bg}; color: {text}; }}
            QGroupBox {{
                background-color: {card};
                border: 1px solid {border};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 2px 6px; color: {accent}; }}
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {accent}, stop:1 {palette['accent_alt']});
                color: white; border: none; padding: 8px 14px; border-radius: 8px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {palette['accent_alt']}, stop:1 {accent});
            }}
            QPushButton:disabled {{
                background: {border}; color: {muted};
            }}
            QLineEdit, QComboBox {{
                background: {input_bg}; border: 1px solid {border}; padding: 6px 8px; border-radius: 8px;
                selection-background-color: {palette['selection_bg']}; selection-color: {palette['selection_text']};
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {accent};
                /* Qt 스타일시트에서는 box-shadow를 지원하지 않으므로 경고를 피하기 위해 제거 */
            }}
            QTableView {{
                background: {card};
                gridline-color: {border};
                alternate-background-color: {palette['table_alt']};
                selection-background-color: {palette['selection_bg']};
                selection-color: {palette['selection_text']};
            }}
            QHeaderView::section {{
                background: {header_bg};
                color: {header_text};
                padding: 6px;
                border: 1px solid {border};
            }}
            QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; padding: 6px; }}
            QTabBar::tab {{ background: {tab}; color: {header_text}; padding: 8px 14px; border-radius: 6px; margin: 2px; }}
            QTabBar::tab:selected {{ background: {palette['tab_selected_bg']}; color: {text}; }}
            QListWidget {{ background: {card}; border: 1px solid {border}; border-radius: 8px; }}
            QListWidget::item:selected {{ background: {palette['selection_bg']}; color: {palette['selection_text']}; }}
            QToolTip {{
                color: {text};
                background-color: {palette['tooltip_bg']};
                border: 1px solid {border};
                padding: 6px;
            }}
            """
        )

    def _theme_palette(self, light: Optional[bool] = None) -> Dict[str, Dict[str, str] | str]:
        is_light = light if light is not None else not getattr(self, "is_dark", True)
        if is_light:
            return {
                "accent": "#6366f1",
                "accent_alt": "#22c55e",
                "bg": "#f8fafc",
                "card": "#ffffff",
                "border": "#e2e8f0",
                "text": "#0f172a",
                "muted": "#475569",
                "input_bg": "#ffffff",
                "header_bg": "#e2e8f0",
                "header_text": "#0f172a",
                "tab_bg": "#e2e8f0",
                "tab_selected_bg": "#ffffff",
                "tooltip_bg": "#ffffff",
                "selection_bg": "#2563eb",
                "selection_text": "#ffffff",
                "table_alt": "#f1f5f9",
                "banner_border": "#cbd5e1",
                "status": {
                    "success": {"bg": "#e3f9e5", "text": "#065f46"},
                    "info": {"bg": "#e0f2fe", "text": "#075985"},
                    "warn": {"bg": "#fff7e0", "text": "#92400e"},
                    "error": {"bg": "#fde8e8", "text": "#7f1d1d"},
                },
                "chart": {
                    "bg": "#ffffff",
                    "grid": "#e2e8f0",
                    "spine": "#cbd5e1",
                    "text": "#0f172a",
                    "legend_face": "#ffffff",
                    "legend_edge": "#e2e8f0",
                    "bull": "#16a34a",
                    "bear": "#ef4444",
                    "ema_fast": "#0ea5e9",
                    "ema_slow": "#a855f7",
                    "ema_long": "#fb923c",
                    "fill": "#2563eb",
                    "trade_buy": "#d97706",
                    "trade_sell": "#22c55e",
                    "heatmap_cmap": "magma",
                    "heatmap_text": "#0f172a",
                    "equity_line": "#16a34a",
                    "supertrend": {"weak": "#d97706", "medium": "#2563eb", "strong": "#ef4444"},
                },
            }

        return {
            "accent": "#7c8bff",
            "accent_alt": "#22d3ee",
            "bg": "#0b1224",
            "card": "#0f172a",
            "border": "#1f2937",
            "text": "#f8fafc",
            "muted": "#cbd5e1",
            "input_bg": "#0b1224",
            "header_bg": "#111827",
            "header_text": "#e2e8f0",
            "tab_bg": "#111827",
            "tab_selected_bg": "#1f2937",
            "tooltip_bg": "#111827",
            "selection_bg": "#1d4ed8",
            "selection_text": "#f8fafc",
            "table_alt": "#0d1428",
            "banner_border": "#23314f",
            "status": {
                "success": {"bg": "#1b4332", "text": "#d1fae5"},
                "info": {"bg": "#12324d", "text": "#cfe4ff"},
                "warn": {"bg": "#4b3212", "text": "#ffe7a3"},
                "error": {"bg": "#4c1d1d", "text": "#fecdd3"},
            },
            "chart": {
                "bg": "#0f172a",
                "grid": "#64748b",
                "spine": "#94a3b8",
                "axis": "#f8fafc",
                "text": "#e2e8f0",
                "legend_face": "#0f172a",
                "legend_edge": "#2c3a52",
                "bull": "#34d399",
                "bear": "#fb7185",
                "ema_fast": "#38bdf8",
                "ema_slow": "#c084fc",
                "ema_long": "#fb923c",
                "fill": "#1d4ed8",
                "trade_buy": "#fbbf24",
                "trade_sell": "#2dd4bf",
                "heatmap_cmap": "magma",
                "heatmap_text": "#e2e8f0",
                "equity_line": "#22d3ee",
                "supertrend": {"weak": "#fbbf24", "medium": "#38bdf8", "strong": "#fb7185"},
            },
        }


class HoldingTableModel(QAbstractTableModel):
    headers = ["종목", "수량", "평단가", "평가액", "손절", "익절", "트레일링"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Dict[str, float | str]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            if index.column() == 0:
                return row.get("market")
            if index.column() == 1:
                return f"{row.get('balance', 0):,.6f}"
            if index.column() == 2:
                return f"{row.get('avg_buy_price', 0):,.0f}"
            if index.column() == 3:
                return f"{row.get('estimated_krw', 0):,.0f}"
            if index.column() == 4:
                return row.get("stop_loss", "-")
            if index.column() == 5:
                return row.get("take_profit", "-")
            if index.column() == 6:
                return row.get("trailing", "-")
        if role == TIMESTAMP_ROLE:
            return row.get("timestamp", datetime.utcnow().timestamp())
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def update(self, holdings) -> None:
        self.beginResetModel()
        self._rows = [
            {
                "market": h.market,
                "balance": h.balance,
                "avg_buy_price": h.avg_buy_price,
                "estimated_krw": h.estimated_krw,
                "stop_loss": "-",
                "take_profit": "-",
                "trailing": "-",
                "timestamp": datetime.utcnow().timestamp(),
            }
            for h in holdings
        ]
        self.endResetModel()


class OrderTableModel(QAbstractTableModel):
    headers = ["시간", "마켓", "사이드", "수량", "가격", "상태"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            values = [
                row.get("time"),
                row.get("market"),
                row.get("side"),
                f"{row.get('volume', 0):,.4f}",
                f"{row.get('price', 0):,.0f}",
                row.get("status"),
            ]
            return values[index.column()]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def add_order(self, order_result) -> None:
        now = datetime.utcnow().strftime("%H:%M:%S")
        status = "완료" if order_result.success else order_result.error or "진행 중"
        self.beginInsertRows(QModelIndex(), 0, 0)
        self._rows.insert(
            0,
            {
                "time": now,
                "market": order_result.market,
                "side": order_result.side,
                "volume": order_result.volume,
                "price": order_result.price,
                "status": status,
            },
        )
        if len(self._rows) > 200:
            self._rows.pop()
        self.endInsertRows()


class TradeHistoryModel(QAbstractTableModel):
    headers = ["시간", "마켓", "신호", "가격", "점수"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Dict[str, Any]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            values = [
                row.get("time"),
                row.get("market"),
                row.get("signal"),
                f"{row.get('price', 0):,.0f}",
                f"{row.get('score', 0):.1f}",
            ]
            return values[index.column()]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def add_trade(self, update: DecisionUpdate) -> None:
        self.beginInsertRows(QModelIndex(), 0, 0)
        self._rows.insert(
            0,
            {
                "time": update.timestamp.strftime("%H:%M:%S"),
                "market": update.market,
                "signal": update.signal.name,
                "price": update.price,
                "score": update.score,
            },
        )
        if len(self._rows) > 200:
            self._rows.pop()
        self.endInsertRows()


class ErrorLogTableModel(QAbstractTableModel):
    headers = ["시간", "레벨", "메시지"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: List[Dict[str, str]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role in (Qt.DisplayRole, Qt.EditRole):
            values = [row.get("time"), row.get("level"), row.get("message")]
            return values[index.column()]
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def add_error(self, message: str, severity: str = "error") -> None:
        now = datetime.utcnow().strftime("%H:%M:%S")
        self.beginInsertRows(QModelIndex(), 0, 0)
        self._rows.insert(0, {"time": now, "level": severity.upper(), "message": message})
        if len(self._rows) > 200:
            self._rows.pop()
        self.endInsertRows()


def main() -> None:
    app = QApplication(sys.argv)
    window = DesktopDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
