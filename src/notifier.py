import requests
import logging

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None):
        """
        Initialize Telegram Notifier.
        :param token: Telegram Bot Token
        :param chat_id: Target Chat ID
        """
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if self.token else None

    def send_alert(self, plate_number: str, vehicle_id: int):
        """
        Send an alert message to Telegram.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not set. Skipping alert.")
            return

        message = f"🚨 **Blacklisted Vehicle Detected!**\n\nPlate: `{plate_number}`\nVehicle ID: {vehicle_id}"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(self.base_url, data=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Telegram alert sent for plate: {plate_number}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
