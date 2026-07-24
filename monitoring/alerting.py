import requests
from loguru import logger
from config.settings import settings
import threading

class TelegramAlerter:
    def __init__(self):
        self.bot_token = getattr(settings, 'TELEGRAM_TOKEN', None)
        self.chat_id = getattr(settings, 'TELEGRAM_CHAT_ID', None)
        self.enabled = bool(self.bot_token and self.chat_id)

    def send_alert(self, message: str, is_async: bool = True):
        if not self.enabled:
            return
            
        def _send():
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"[Telegram] Failed to send alert: {response.text}")
            except Exception as e:
                logger.error(f"[Telegram] Error sending alert: {e}")
                
        if is_async:
            threading.Thread(target=_send, daemon=True).start()
        else:
            _send()

telegram_alerter = TelegramAlerter()
