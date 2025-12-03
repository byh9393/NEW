"""
SQLite 기반 상태 저장/복구 및 DB 스키마 정의.

- 계좌 스냅샷, 포지션, 주문/체결, 전략 상태, 리스크 이벤트 로그를 저장해 재시작 시 복구에 활용
- 업비트 재시작 절차(/v1/accounts, /v1/orders 동기화) 전에 로컬 스냅샷으로 빠른 UI 부팅 지원
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from upbit_bot.trading.account import AccountSnapshot, Holding
from upbit_bot.trading.executor import OrderResult


@dataclass
class PositionRecord:
    market: str
    volume: float
    avg_price: float
    opened_at: Optional[datetime] = None


class SQLiteStateStore:
    def __init__(self, db_path: str | Path = "./.state/trading.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row

    def ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                krw_balance REAL,
                total_value REAL,
                total_fee REAL,
                profit REAL,
                profit_pct REAL
            );

            CREATE TABLE IF NOT EXISTS positions (
                market TEXT PRIMARY KEY,
                volume REAL,
                avg_price REAL,
                opened_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT,
                market TEXT,
                side TEXT,
                price REAL,
                volume REAL,
                status TEXT,
                raw TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_uuid TEXT,
                market TEXT,
                side TEXT,
                price REAL,
                volume REAL,
                fee REAL,
                net_amount REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS candles (
                market TEXT,
                timeframe TEXT,
                ts TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (market, timeframe, ts)
            );

            CREATE TABLE IF NOT EXISTS strategy_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market TEXT,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self.conn.commit()

    def persist_snapshot(self, snapshot: AccountSnapshot) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO accounts_snapshot (krw_balance, total_value, total_fee, profit, profit_pct)
            VALUES (?, ?, ?, ?, ?)
            """,
            (snapshot.krw_balance, snapshot.total_value, snapshot.total_fee, snapshot.profit, snapshot.profit_pct),
        )
        self._upsert_positions(snapshot.holdings)
        self.conn.commit()

    def record_order(self, *, order_result: OrderResult) -> None:
        cur = self.conn.cursor()
        raw = json.dumps(order_result.raw, ensure_ascii=False)
        uuid = None
        if isinstance(order_result.raw, dict):
            attempts = order_result.raw.get("attempts") if order_result.raw else None
            if attempts and isinstance(attempts, list) and attempts:
                uuid = attempts[0].get("uuid") if isinstance(attempts[0], dict) else None
        cur.execute(
            """
            INSERT INTO orders (uuid, market, side, price, volume, status, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid,
                order_result.market,
                order_result.side,
                order_result.price,
                order_result.volume,
                "done" if order_result.success else "rejected",
                raw,
            ),
        )
        if order_result.success:
            cur.execute(
                """
                INSERT INTO trades (order_uuid, market, side, price, volume, fee, net_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid,
                    order_result.market,
                    order_result.side,
                    order_result.price,
                    order_result.volume,
                    order_result.fee,
                    order_result.net_amount,
                ),
            )
        self.conn.commit()

    def record_risk_event(self, *, market: str, reason: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO risk_events (market, reason) VALUES (?, ?)",
            (market, reason),
        )
        self.conn.commit()

    def persist_strategy_state(self, payload: dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO strategy_state (id, payload, updated_at)
            VALUES (1, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (json.dumps(payload, ensure_ascii=False),),
        )
        self.conn.commit()

    def load_positions(self) -> List[PositionRecord]:
        cur = self.conn.cursor()
        cur.execute("SELECT market, volume, avg_price, opened_at FROM positions WHERE volume > 0")
        rows = cur.fetchall()
        return [
            PositionRecord(
                market=row["market"],
                volume=float(row["volume"] or 0.0),
                avg_price=float(row["avg_price"] or 0.0),
                opened_at=self._parse_dt(row["opened_at"]),
            )
            for row in rows
        ]

    def save_config(self, payload: dict) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO config_history (payload) VALUES (?)",
            (json.dumps(payload, ensure_ascii=False),),
        )
        self.conn.commit()

    def _upsert_positions(self, holdings: Iterable[Holding]) -> None:
        cur = self.conn.cursor()
        for holding in holdings:
            cur.execute(
                """
                INSERT INTO positions (market, volume, avg_price, opened_at, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(market) DO UPDATE SET
                    volume=excluded.volume,
                    avg_price=excluded.avg_price,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (holding.market, holding.balance, holding.avg_buy_price),
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _parse_dt(value: object) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, (bytes, bytearray)):
            value = value.decode()
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
