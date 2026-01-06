"""
SMS Sender Module
Sends SMS alerts via Twilio
"""
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import TWILIO
from src.utils.logger import setup_logger

logger = setup_logger('SMSSender')


class SMSSender:
    """
    Sends SMS alerts using Twilio API.
    """
    
    def __init__(self):
        """Initialize Twilio client."""
        self.account_sid = TWILIO['account_sid']
        self.auth_token = TWILIO['auth_token']
        self.from_number = TWILIO['phone_number']
        self.recipient_numbers = TWILIO['recipient_numbers']
        
        if not self.account_sid or not self.auth_token:
            logger.warning("Twilio credentials not configured - SMS disabled")
            self.enabled = False
            return
        
        try:
            from twilio.rest import Client
            self.client = Client(self.account_sid, self.auth_token)
            self.enabled = True
            logger.info("SMS sender initialized successfully")
        except ImportError:
            logger.error("Twilio library not installed - SMS disabled")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Twilio: {e}")
            self.enabled = False
    
    def send_sms(self, message: str, recipients: List[str] = None) -> bool:
        """
        Send SMS to recipients.
        
        Args:
            message: Message text
            recipients: List of phone numbers (optional, uses default if None)
            
        Returns:
            True if all messages sent successfully
        """
        if not self.enabled:
            logger.info(f"SMS SIMULATION: {message}")
            return False
        
        if recipients is None:
            recipients = self.recipient_numbers
        
        if not recipients:
            logger.warning("No recipient numbers configured")
            return False
        
        success = True
        
        for recipient in recipients:
            try:
                message_obj = self.client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=recipient
                )
                logger.info(f"SMS sent to {recipient}: {message_obj.sid}")
            except Exception as e:
                logger.error(f"Failed to send SMS to {recipient}: {e}")
                success = False
        
        return success
    
    def send_alert_sms(self, alert_data: dict) -> bool:
        """
        Send formatted alert SMS.
        
        Args:
            alert_data: Alert information dictionary
            
        Returns:
            True if successful
        """
        priority = alert_data.get('priority', 'UNKNOWN')
        alert_type = alert_data.get('type', 'UNKNOWN')
        camera = alert_data.get('camera_id', 'N/A')
        timestamp = alert_data.get('timestamp', '')
        
        message = f"🚨 [{priority}] {alert_type}\n"
        message += f"Camera: {camera}\n"
        message += f"Time: {timestamp}\n"
        message += "Check app for details."
        
        return self.send_sms(message)
