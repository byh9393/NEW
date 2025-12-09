"""
FastAPI-based dashboard backend.

- Exposes REST endpoints for account snapshot, positions, orders, and risk events.
- Provides a lightweight WebSocket stream that emits the latest snapshot for monitoring.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, WebSocket
from fastapi.responses import JSONResponse

from upbit_bot.storage import SQLiteStateStore


def _store(db_path: str | Path = "./.state/trading.db") -> SQLiteStateStore:
    store = SQLiteStateStore(db_path=db_path)
    store.ensure_schema()
    return store


def get_app(db_path: str | Path = "./.state/trading.db") -> FastAPI:
    app = FastAPI(title="Upbit Bot Dashboard", version="0.1.0")

    def store_dep() -> SQLiteStateStore:
        return _store(db_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/account")
    def account(store: SQLiteStateStore = Depends(store_dep)) -> JSONResponse:
        snap = store.load_latest_snapshot()
        if not snap:
            return JSONResponse({"snapshot": None})
        payload = {
            "krw_balance": snap.krw_balance,
            "total_value": snap.total_value,
            "total_fee": snap.total_fee,
            "profit": snap.profit,
            "profit_pct": snap.profit_pct,
            "holdings": [
                {
                    "market": h.market,
                    "currency": h.currency,
                    "balance": h.balance,
                    "avg_buy_price": h.avg_buy_price,
                    "estimated_krw": h.estimated_krw,
                }
                for h in snap.holdings
            ],
        }
        return JSONResponse({"snapshot": payload})

    @app.get("/positions")
    def positions(store: SQLiteStateStore = Depends(store_dep)) -> dict:
        pos = store.load_positions()
        return {
            "positions": [
                {
                    "market": p.market,
                    "volume": p.volume,
                    "avg_price": p.avg_price,
                    "opened_at": p.opened_at.isoformat() if p.opened_at else None,
                }
                for p in pos
            ]
        }

    @app.get("/orders")
    def orders(limit: int = 50, store: SQLiteStateStore = Depends(store_dep)) -> dict:
        rows = store.load_recent_orders(limit=limit)
        return {"orders": [dict(row) for row in rows]}

    @app.get("/risk-events")
    def risk_events(limit: int = 50, store: SQLiteStateStore = Depends(store_dep)) -> dict:
        rows = store.load_risk_events(limit=limit)
        return {"events": [dict(row) for row in rows]}

    @app.websocket("/ws/stream")
    async def stream(ws: WebSocket, store: SQLiteStateStore = Depends(store_dep)) -> None:
        await ws.accept()
        snap = store.load_latest_snapshot()
        payload: Optional[dict] = None
        if snap:
            payload = {
                "krw_balance": snap.krw_balance,
                "total_value": snap.total_value,
                "profit_pct": snap.profit_pct,
                "holdings": [
                    {"market": h.market, "balance": h.balance, "estimated_krw": h.estimated_krw} for h in snap.holdings
                ],
            }
        await ws.send_json({"type": "snapshot", "data": payload})
        # Keep alive briefly to allow client consumption
        try:
            for _ in range(3):
                await asyncio.sleep(1)
                await ws.send_json({"type": "ping"})
        except Exception:
            pass
        await ws.close()

    return app


app = get_app()
