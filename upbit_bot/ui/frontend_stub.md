Frontend wiring guide (to be implemented in your React/Tailwind app)
------------------------------------------------------------------

API base: http://localhost:8000

Endpoints:
- GET /health -> {status}
- GET /account -> {snapshot: {krw_balance,total_value,total_fee,profit,profit_pct,holdings[]}}
- GET /positions -> {positions:[{market,volume,avg_price,opened_at}]}
- GET /orders -> {orders:[...]}
- GET /risk-events -> {events:[...]}
- GET /status -> {snapshot,strategy_state,risk_events}
- POST /controls/global {enabled: bool}
- POST /controls/emergency {active: bool, close_positions: bool}
- POST /controls/strategy {name, enabled}
- POST /config {...} (persist history)
- WS /ws/stream -> {"type":"snapshot","data":{...}} + periodic {"type":"ping"}

Suggested UI sections (matching AGENTS.md):
- Header cards: total_value, krw_balance, profit_pct, daily PnL (derive from snapshot history).
- Controls: global ON/OFF, emergency stop (with confirm), strategy toggles (trend/mean_reversion/breakout).
- Positions table: market, size, avg, last price, PnL%, stops (future).
- Orders & risk events tabs: recent orders, risk logs.
- Heatmap: use /account holdings + price buffer (future WS) to render momentum/trend scores when exposed.
- PnL chart: poll /account snapshots (or add /pnl endpoint) and render cumulative curve.

Data polling cadence:
- Snapshot/status: 3–5s poll + WS push.
- Orders/risk: 10–15s poll.

Auth: (add when ready) include bearer/JWT if needed; for now, none.

Next steps to implement:
- Build React query hooks for the above routes.
- Add chart components (Recharts/Apex) for equity curve and heatmap once backend exposes factors.
