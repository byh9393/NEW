"""
FastAPI-based dashboard backend.

- Exposes REST endpoints for account snapshot, positions, orders, and risk events.
- Provides a lightweight WebSocket stream that emits the latest snapshot for monitoring.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Any, Dict

from fastapi import Depends, FastAPI, WebSocket
from fastapi.responses import JSONResponse

from upbit_bot.storage import SQLiteStateStore


def _store(db_path: str | Path = "./.state/trading.db") -> SQLiteStateStore:
    store = SQLiteStateStore(db_path=db_path)
    store.ensure_schema()
    return store


def get_app(db_path: str | Path = "./.state/trading.db", bot: Any = None) -> FastAPI:
    app = FastAPI(title="Upbit Bot Dashboard", version="0.2.0")

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

    @app.get("/status")
    def status(store: SQLiteStateStore = Depends(store_dep)) -> dict:
        snap = store.load_latest_snapshot()
        strategy_state = store.load_strategy_state()
        return {
            "snapshot": None
            if not snap
            else {
                "krw_balance": snap.krw_balance,
                "total_value": snap.total_value,
                "profit_pct": snap.profit_pct,
            },
            "strategy_state": strategy_state,
            "risk_events": store.load_risk_events(limit=10),
        }

    @app.get("/pnl")
    def pnl(limit: int = 500, store: SQLiteStateStore = Depends(store_dep)) -> dict:
        curve = store.load_equity_curve(limit=limit)
        return {"equity_curve": curve}

    @app.get("/heatmap")
    def heatmap() -> dict:
        # Placeholder: frontend can render empty until factors exposed
        return {"heatmap": []}

    @app.post("/controls/global")
    def toggle_global(payload: Dict[str, bool], store: SQLiteStateStore = Depends(store_dep)) -> dict:
        enabled = bool(payload.get("enabled", True))
        if bot:
            try:
                bot.set_global_enabled(enabled)
            except Exception:
                pass
        state = store.load_strategy_state() or {}
        state["global_enabled"] = enabled
        store.persist_strategy_state(state)
        return {"global_enabled": enabled}

    @app.post("/controls/emergency")
    def emergency(payload: Dict[str, bool], store: SQLiteStateStore = Depends(store_dep)) -> dict:
        active = bool(payload.get("active", False))
        close_positions = bool(payload.get("close_positions", False))
        if bot:
            try:
                bot.set_emergency_stop(active=active, close_positions=close_positions)
            except Exception:
                pass
        state = store.load_strategy_state() or {}
        state["emergency_stop"] = {"active": active, "close_positions": close_positions}
        store.persist_strategy_state(state)
        return state["emergency_stop"]

    @app.post("/controls/strategy")
    def strategy_switch(payload: Dict[str, Any], store: SQLiteStateStore = Depends(store_dep)) -> dict:
        name = str(payload.get("name", ""))
        enabled = bool(payload.get("enabled", True))
        if bot and name:
            try:
                bot.set_strategy_enabled(name, enabled)
            except Exception:
                pass
        state = store.load_strategy_state() or {}
        strategies = state.get("strategies", {})
        strategies[name] = enabled
        state["strategies"] = strategies
        store.persist_strategy_state(state)
        return {"name": name, "enabled": enabled}

    @app.post("/config")
    def save_config(payload: Dict[str, Any], store: SQLiteStateStore = Depends(store_dep)) -> dict:
        store.save_config(payload)
        return {"saved": True}

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
