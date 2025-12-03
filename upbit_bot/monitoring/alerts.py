"""
Slack/Telegram 알림 모듈.

- 리스크 이벤트, 주문 거절, 데이터 오류 등을 웹훅으로 통지
- 환경 변수 기반으로 비활성화 가능(기본 비활성)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertMessage:
    title: str
    detail: str
    severity: Severity
    context: Optional[Dict[str, str]] = None


class AlertSink:
    def __init__(self) -> None:
        self.slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    def notify(self, alert: AlertMessage) -> None:
        sent = False
        if self.slack_webhook:
            sent = self._post_slack(alert) or sent
        if self.telegram_token and self.telegram_chat_id:
            sent = self._post_telegram(alert) or sent
        if not sent:
            logger.info("[알림 생략] %s - %s", alert.severity.value, alert.title)

    def _post_slack(self, alert: AlertMessage) -> bool:
        payload = {
            "text": f"[{alert.severity.value.upper()}] {alert.title}\n{alert.detail}",
        }
        try:
            resp = requests.post(self.slack_webhook, json=payload, timeout=5)
            resp.raise_for_status()
            return True
        except Exception:
            logger.exception("Slack 알림 전송 실패")
            return False

    def _post_telegram(self, alert: AlertMessage) -> bool:
        api_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        text = f"[{alert.severity.value.upper()}] {alert.title}\n{alert.detail}"
        try:
            resp = requests.post(api_url, data={"chat_id": self.telegram_chat_id, "text": text}, timeout=5)
            resp.raise_for_status()
            return True
        except Exception:
            logger.exception("Telegram 알림 전송 실패")
            return False
