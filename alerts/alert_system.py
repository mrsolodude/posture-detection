from twilio.rest import Client
import logging
import os

# Configure your Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "ACxxxxxx")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "your_token")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "+1234567890")

class AlertSystem:
    def __init__(self, account_sid=TWILIO_ACCOUNT_SID, auth_token=TWILIO_AUTH_TOKEN, from_number=TWILIO_PHONE_NUMBER):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.client = Client(account_sid, auth_token) if account_sid != "ACxxxxxx" else None

    def send_sms(self, to_number, message):
        """Sends an SMS using Twilio."""
        try:
            if not self.client:
                logging.info(f"[SIMULATED SMS to {to_number}]: {message}")
                return "Simulated success"
            
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            return message.sid
        except Exception as e:
            logging.error(f"Error sending SMS: {e}")
            return None

# Singleton instance
alert_system = AlertSystem()
