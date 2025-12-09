"""
Alert hooks (Telegram stub).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str


class AlertSink:
    def __init__(self, telegram: Optional[TelegramConfig] = None) -> None:
        self.telegram = telegram

    def notify(self, title: str, message: str) -> None:
        if self.telegram:
            self._send_telegram(f"{title}\n{message}")

    def _send_telegram(self, text: str) -> None:
        if not self.telegram:
            return
        url = f"https://api.telegram.org/bot{self.telegram.bot_token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.telegram.chat_id, "text": text}, timeout=5)
        except Exception:
            logger.exception("Telegram alert failed")
