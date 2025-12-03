# 업비트 자동매매 봇

Python 기반으로 업비트 거래소 모든 원화마켓을 실시간 추적하며 **OpenAI 기반 평가**와 다중 기술적 지표(EMA 트렌드, RSI·Stochastic 모멘텀, 볼린저/켈트너 변동성, 평균회귀·품질 필터)를 조합해 매수·매도 타이밍을 판단하는 자동매매 프로젝트입니다. 기본값은 모의주문 모드로 실행되며, 실제 주문을 사용하려면 API 키를 환경 변수로 설정하면 됩니다.

## 특징
- **시장 수집**: 업비트 REST API로 모든 KRW 마켓 리스트를 받아옵니다.
- **Upbit 어댑터**: `UpbitAdapter`가 Remaining-Req 헤더를 해석해 초당 한도 여유가 적을 때 자동 대기하고, 429/5xx 응답에는 지수형 백오프로 재시도합니다.
- **실시간 데이터**: 업비트 웹소켓을 통해 틱 가격을 수신하고 버퍼에 저장합니다.
- **기술적 지표**: EMA(20/60/200) 정렬·기울기, RSI/스토캐스틱 레벨·기울기, 단/중기 ROC 및 가속도, 볼린저 밴드 위치·밴드폭과 **켈트너 스퀴즈 필터**, Z-Score, Choppiness 기반 시장 품질 필터 등 시스템 트레이더가 선호하는 다중 지표를 계산합니다.
- **전략**: 추세·모멘텀·변동성·평균회귀 네 축을 가중 합산한 복합 점수(-100~100)에 따라 매수/매도/대기 신호를 생성합니다.
- **OpenAI 강화 판단**: 기본 전략 결과와 확장된 지표 요약을 GPT-4o-mini 등에 전달해 전문가형 BUY/SELL/HOLD 결정을 JSON으로 회신받아 실행합니다.
- **주문 실행**: 환경 변수 `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY`가 없을 경우 모의주문으로 안전하게 동작합니다.
- **계좌 모니터링**: GUI에서 실시간 원화 잔고와 보유 종목, 평가액을 확인할 수 있습니다.
- **상태 저장/복구**: SQLite 로컬 DB에 계좌 스냅샷·포지션·주문/체결·리스크 이벤트·설정 변경 내역을 기록하여 재시작 시 포지션과 평균단가를 즉시 복구합니다.
- **백테스트 엔진**: 실거래와 동일한 전략/리스크 로직을 공유하는 `BacktestEngine`을 통해 수수료·슬리피지·최소 주문금액을 반영한 포트폴리오 시뮬레이션과 MDD/Sharpe/Profit Factor 지표를 산출합니다.
- **동적 유니버스 관리**: 30일 평균 일 거래대금이 기준치 이상이고 호가 스프레드가 좁은 종목만 남기며, 24시간 거래대금 상위 N개로 자동 제한합니다.
- **알림**: Slack/Telegram 웹훅으로 리스크 이벤트나 주문 거절을 통지할 수 있으며, 환경 변수로 손쉽게 끄고 켤 수 있습니다.

## 설치
```bash
pip install -r requirements.txt
```

## 시작 가이드 (단계별)
1. **필수 준비물 확인**
   - Python 3.10 이상과 `pip`이 설치되어 있어야 합니다.
   - 실거래를 사용할 계획이면 업비트 OPEN API 발급을 완료하고, OpenAI 판단을 켜려면 `OPENAI_API_KEY`를 준비합니다.

2. **가상환경 생성(권장)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

3. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4. **환경 변수(.env) 작성**
   - 저장소 루트에 `.env`를 만들고 아래 예시를 채웁니다. 키가 없으면 모의주문으로만 실행됩니다.
     ```env
     OPENAI_API_KEY=your-openai-api-key
     UPBIT_ACCESS_KEY=your-upbit-access-key
     UPBIT_SECRET_KEY=your-upbit-secret-key
     MIN_30D_AVG_TURNOVER=1000000000
     MAX_SPREAD_PCT=2.0
     SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
     TELEGRAM_BOT_TOKEN=token
     TELEGRAM_CHAT_ID=12345678
     ```
   - `.env`를 작성하지 않더라도 실행은 가능하지만, 실거래·알림·LLM 판단 기능은 비활성화됩니다.

5. **데이터베이스 설정**
   - 기본 저장소는 프로젝트 루트 하위 `./.state/trading.db` SQLite 파일이며, 봇을 최초 실행하면 자동 생성·스키마 마이그레이션(`ensure_schema`)이 수행됩니다.
   - 저장 경로를 바꾸려면 `TradingBot(state_store=SQLiteStateStore(db_path="./data/trading.db"))`처럼 직접 경로를 넘겨 실행하세요.
   - Docker·서버 환경에서는 DB 디렉터리를 볼륨으로 마운트해 컨테이너 재시작 시 상태(포지션, 주문·체결, 전략/리스크 이벤트)가 유지되도록 합니다.

6. **데모 실행(모의주문 기본값)**
   - 10초 동안 실시간 가격을 구독하며 신호를 출력합니다.
   ```bash
   python -m upbit_bot.app
   ```
   - 기본값은 `simulated=True`로 모의 체결만 발생하며, 계좌·주문 내역은 SQLite에 기록됩니다.

7. **실거래로 전환(선택)**
   - `.env`에 업비트 키를 설정한 뒤, 실행 시 `TRADING_MODE=live` 환경 변수를 추가하거나 `TradingBot(simulated=False)`로 생성하도록 설정합니다.
   - 업비트 최소 주문금액 5,000원과 봇 내부 최소 30,000원 중 더 큰 값이 적용되므로, 잔고를 충분히 확보한 뒤 실거래를 시작하세요.

8. **GUI 대시보드 열기(Pyside6)**
   - 거래 시작/종료, 모의주문·OpenAI 판단 토글, 실시간 신호·가격·계좌·로그 확인 기능을 제공합니다.
   ```bash
   python -m upbit_bot.ui.desktop
   ```
   - 컬럼 정렬·필터·검색, 다크/라이트 테마 전환, 주문/에러 알림 배너, 즐겨찾기 핀, 계좌/차트/로그 패널을 한 화면에서 사용할 수 있습니다.

9. **백테스트 실행**
   - OHLCV CSV를 준비한 뒤 예시 코드를 실행해 전략/리스크 로직을 검증합니다(아래 예시 참조).
   - 결과로 MDD/Sharpe/Profit Factor 등 통계가 출력되며, 실거래와 동일한 수수료·슬리피지·최소 주문금액 로직이 적용됩니다.

10. **로그·데이터 확인 및 종료**
   - 로그와 SQLite DB는 기본적으로 프로젝트 루트의 `upbit_bot` 하위 또는 실행 디렉터리에 생성됩니다.
   - 모니터링을 끝냈다면 터미널에서 `Ctrl+C`로 종료하고, 실거래 모드였다면 대시보드에서 전략 OFF 또는 “긴급 ALL STOP” 버튼으로 진입을 막은 뒤 종료합니다.

## 데이터베이스 설정 상세
- **기본 경로**: `SQLiteStateStore`가 `./.state/trading.db`를 사용하며, 실행 시 자동 생성합니다.
- **스키마**: 계좌 스냅샷, 포지션, 주문/체결, 캔들, 전략 상태, 리스크 이벤트, 설정 변경 내역 테이블을 포함합니다.
- **경로 변경**: 직접 인스턴스화할 때 `SQLiteStateStore(db_path="/var/lib/upbit_bot/trading.db")`로 지정하고, 백업·복구 시 해당 파일만 보존하면 됩니다.
- **백업 팁**: 서비스 중단 전에 파일을 복사하거나, 백업 전 `trading_bot.stop_event`를 설정해 트레이딩 루프를 멈춘 뒤 DB를 백업하면 일관성이 높습니다.

## 백테스트 실행 예시
`BacktestEngine`으로 다중 종목 OHLCV를 검증할 수 있습니다.

```python
import pandas as pd
from upbit_bot.backtest import BacktestEngine, BacktestConfig

data = {
    "KRW-BTC": pd.read_csv("btc_5m.csv", parse_dates=["timestamp"], index_col="timestamp"),
    "KRW-ETH": pd.read_csv("eth_5m.csv", parse_dates=["timestamp"], index_col="timestamp"),
}

engine = BacktestEngine(BacktestConfig(initial_cash=2_000_000, slippage_pct=0.05))
result = engine.run(data)
print(result.stats)
```

PySide6 기반 카드형 그리드 UI 데스크톱 대시보드를 이용해 컬럼 정렬·필터·검색, 다크/라이트 테마 전환, 주문·에러 알림 배너, 즐겨찾기 핀, 계좌/차트/로그를 한 화면에서 확인할 수 있습니다.
```bash
python -m upbit_bot.ui.desktop
```

## 환경 변수 및 .env 설정
- 프로젝트 루트의 `.env` 파일에 아래 키를 입력하면 패키지 초기화 시 자동으로 불러옵니다.
  ```env
  OPENAI_API_KEY=your-openai-api-key
  UPBIT_ACCESS_KEY=your-upbit-access-key
  UPBIT_SECRET_KEY=your-upbit-secret-key
  MIN_30D_AVG_TURNOVER=1000000000  # 30일 평균 일 거래대금 하한
  MAX_SPREAD_PCT=2.0               # 허용 스프레드(%)
  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
  TELEGRAM_BOT_TOKEN=token
  TELEGRAM_CHAT_ID=12345678
  ```
- `OPENAI_API_KEY`: OpenAI 모델 호출에 사용. 설정되어 있으면 LLM이 기본 기술적 판단을 보강합니다.
- `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY`: 실거래 주문용 업비트 API 키. 둘 다 없으면 모의주문으로 실행합니다.
 - `MIN_30D_AVG_TURNOVER`, `MAX_SPREAD_PCT`: 유동성/스프레드 필터 기준.
 - `SLACK_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`: 알림 채널 설정.

## 실제 주문 사용 시 주의
- 실거래를 활성화하려면 환경 변수에 업비트 API 키를 설정하고 `TradingBot(simulated=False)`로 생성하거나 `UPBIT_ACCESS_KEY`, `UPBIT_SECRET_KEY`를 주입하세요.
- 자동매매는 원금 손실 위험이 있습니다. 본 코드는 참고용이며, 실제 자금 운용 전 충분한 테스트와 리스크 관리가 필요합니다.
- 업비트는 5,000원 미만 주문이 불가능하며, 본 봇은 안전을 위해 **실제 최소 주문금액 = max(봇 내부 30,000원, 거래소 min_total)**을 적용합니다. 매수 시 신호 강도를 반영해 원화 잔고 대비 5~25% 비중으로 자동 배분합니다.
