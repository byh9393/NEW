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
from typing import Deque, Dict, Iterable, List, Optional, Set

from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QObject, Qt,
                            QSortFilterProxyModel, QTimer, Signal)
from PySide6.QtGui import QColor, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from upbit_bot.app import DecisionUpdate, TradingBot
from upbit_bot.data.market_fetcher import fetch_markets

PIN_ROLE = Qt.UserRole + 1
TIMESTAMP_ROLE = Qt.UserRole + 2


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
        else:
            row = self._order.index(market)
            top_left = self.index(row, 0)
            bottom_right = self.index(row, len(self.headers) - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.DisplayRole, Qt.EditRole])


class SignalFilterProxy(QSortFilterProxyModel):
    def __init__(self) -> None:
        super().__init__()
        self._query = ""

    def set_query(self, text: str) -> None:
        self._query = text.lower()
        self.invalidateFilter()

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

        self.adapter = UpdateAdapter()
        self.runner = BotRunner(self.adapter.on_update)
        self.adapter.update_received.connect(self._handle_update)
        self.adapter.start()

        self.signal_model = SignalTableModel()
        self.proxy_model = SignalFilterProxy()
        self.proxy_model.setSourceModel(self.signal_model)
        self.price_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
        self.latest_account = None
        self.active_markets: List[str] = []
        self._chart_dirty = False
        self._chart_timer = QTimer(self)
        self._chart_timer.setInterval(800)
        self._chart_timer.timeout.connect(self._refresh_chart_if_needed)
        self._chart_timer.start()
        self._max_log_lines = 500

        self._init_ui()
        self._apply_light_theme()

    # UI 구성
    def _init_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self._build_toolbar())
        layout.addWidget(self._build_banner())

        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(8)

        grid_layout.addWidget(self._build_signals_card(), 0, 0, 2, 2)
        grid_layout.addWidget(self._build_account_card(), 0, 2, 1, 1)
        grid_layout.addWidget(self._build_chart_card(), 1, 2, 1, 1)
        grid_layout.addWidget(self._build_log_card(), 2, 0, 1, 3)

        layout.addWidget(grid_container)
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
        self.banner.setStyleSheet("QLabel { background:#f0f0f0; border-radius:6px; padding:6px; }")
        return self.banner

    def _build_signals_card(self) -> QWidget:
        box = QGroupBox("신호 테이블")
        v = QVBoxLayout(box)
        self.signal_view = QTableView()
        self.signal_view.setModel(self.proxy_model)
        self.signal_view.setSortingEnabled(True)
        self.signal_view.sortByColumn(7, Qt.DescendingOrder)
        self.signal_view.clicked.connect(self._handle_table_click)
        self.signal_view.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.signal_view)
        return box

    def _build_account_card(self) -> QWidget:
        box = QGroupBox("계좌 / 즐겨찾기")
        v = QVBoxLayout(box)

        self.balance_label = QLabel("원화 잔고: -")
        self.total_label = QLabel("총 평가액: -")
        v.addWidget(self.balance_label)
        v.addWidget(self.total_label)

        self.favorite_box = QComboBox()
        self.favorite_box.setPlaceholderText("즐겨찾기 마켓")
        v.addWidget(self.favorite_box)

        self.holding_table = QTableView()
        self.holding_model = HoldingTableModel()
        self.holding_view_proxy = QSortFilterProxyModel()
        self.holding_view_proxy.setSourceModel(self.holding_model)
        self.holding_view_proxy.setSortRole(TIMESTAMP_ROLE)
        self.holding_table.setModel(self.holding_view_proxy)
        self.holding_table.setSortingEnabled(True)
        self.holding_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.holding_table)
        return box

    def _build_chart_card(self) -> QWidget:
        box = QGroupBox("차트")
        v = QVBoxLayout(box)
        top = QHBoxLayout()
        top.addWidget(QLabel("대상 마켓"))
        self.chart_selector = QComboBox()
        self.chart_selector.currentTextChanged.connect(self._refresh_chart)
        top.addWidget(self.chart_selector)
        top.addStretch(1)
        v.addLayout(top)

        fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v.addWidget(self.canvas)
        return box

    def _build_log_card(self) -> QWidget:
        box = QGroupBox("로그")
        v = QVBoxLayout(box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)
        return box

    # 이벤트 처리
    def start_trading(self) -> None:
        text = self.market_input.text().strip()
        if text:
            markets = [m.strip().upper() for m in text.split(",") if m.strip()]
        else:
            markets = fetch_markets(is_fiat=True, fiat_symbol="KRW")

        if not markets:
            QMessageBox.warning(self, "마켓 없음", "구독할 마켓을 찾지 못했습니다.")
            return

        self.runner.start(markets, simulated=self.simulated_check.isChecked(), use_ai=self.ai_check.isChecked())
        self.active_markets = markets
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._show_banner(f"{len(markets)}개 마켓 구독 시작", success=True)
        self.status_label.setText(f"실행 중 | 모니터링 {len(markets)}개")

    def stop_trading(self) -> None:
        self.runner.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._show_banner("거래 종료", success=True)
        self.status_label.setText("대기 중")
        self.active_markets = []

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
        self.proxy_model.invalidateFilter()
        self.price_history[update.market].append(update.price)
        if self.chart_selector.findText(update.market) == -1:
            self.chart_selector.addItem(update.market)
        if not self.chart_selector.currentText():
            self.chart_selector.setCurrentText(update.market)
        if update.market == self.chart_selector.currentText():
            self._chart_dirty = True

        if update.account:
            self.latest_account = update.account
            self._update_account(update.account)

        self._append_log(update)

        if update.executed:
            self._show_banner(f"{update.market} {update.signal.name} 주문 처리", success=True)
        if update.order_result and update.order_result.error:
            self._show_banner(update.order_result.error, success=False)

    def _append_log(self, update: DecisionUpdate) -> None:
        ts = update.timestamp.strftime("%H:%M:%S")
        ai_txt = (update.ai_raw or "-").split("\n")[0]
        ai_short = ai_txt[:80] + ("..." if len(ai_txt) > 80 else "")
        status = "실행" if update.executed else "대기"
        line = f"[{ts}] {update.market} {update.signal.name} 점수 {update.score:.1f} ({status}) | {update.reason} | AI={ai_short}\n"
        self.log_view.append(line)
        if self.log_view.document().blockCount() > self._max_log_lines:
            cursor = self.log_view.textCursor()
            cursor.movePosition(QTextCursor.Start)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _update_account(self, snapshot) -> None:
        self.balance_label.setText(f"원화 잔고: {snapshot.krw_balance:,.0f}원")
        self.total_label.setText(f"총 평가액: {snapshot.total_value:,.0f}원")
        self.holding_model.update(snapshot.holdings)
        self.holding_view_proxy.invalidate()

    def _refresh_chart(self) -> None:
        market = self.chart_selector.currentText()
        self.ax.clear()
        self.ax.set_title(market or "마켓 선택")
        if market and market in self.price_history:
            self.ax.plot(list(self.price_history[market]), color="tab:blue")
        self.ax.set_xlabel("틱")
        self.ax.set_ylabel("가격")
        self.canvas.draw_idle()

    def _refresh_chart_if_needed(self) -> None:
        if not self._chart_dirty:
            return
        self._chart_dirty = False
        self._refresh_chart()

    def _show_banner(self, text: str, *, success: bool) -> None:
        color = "#d1f2d9" if success else "#ffd6d6"
        self.banner.setText(text)
        self.banner.setStyleSheet(f"QLabel {{ background:{color}; border-radius:6px; padding:6px; }}")

    def _toggle_theme(self, checked: bool) -> None:
        if checked:
            self._apply_dark_theme()
            self.theme_btn.setText("다크")
        else:
            self._apply_light_theme()
            self.theme_btn.setText("라이트")

    def _apply_dark_theme(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(palette)

    def _apply_light_theme(self) -> None:
        app = QApplication.instance()
        if not app:
            return
        app.setPalette(QApplication.style().standardPalette())


class HoldingTableModel(QAbstractTableModel):
    headers = ["종목", "수량", "평단가", "평가액"]

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
                "timestamp": datetime.utcnow().timestamp(),
            }
            for h in holdings
        ]
        self.endResetModel()


def main() -> None:
    app = QApplication(sys.argv)
    window = DesktopDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
