"""
Push Notification Module
Sends push notifications via Firebase Cloud Messaging
"""
import sys
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import FIREBASE
from src.utils.logger import setup_logger

logger = setup_logger('PushNotifier')


class PushNotifier:
    """
    Sends push notifications using Firebase Cloud Messaging.
    """
    
    def __init__(self):
        """Initialize Firebase admin SDK."""
        self.credentials_path = FIREBASE['credentials_path']
        self.topic = FIREBASE['topic']
        
        if not Path(self.credentials_path).exists():
            logger.warning(f"Firebase credentials not found at {self.credentials_path}")
            self.enabled = False
            return
        
        try:
            import firebase_admin
            from firebase_admin import credentials, messaging
            
            cred = credentials.Certificate(self.credentials_path)
            firebase_admin.initialize_app(cred)
            
            self.messaging = messaging
            self.enabled = True
            logger.info("Push notifier initialized successfully")
            
        except ImportError:
            logger.error("Firebase admin SDK not installed - push notifications disabled")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            self.enabled = False
    
    def send_notification(self, title: str, body: str, 
                        data: Dict = None) -> bool:
        """
        Send push notification to topic.
        
        Args:
            title: Notification title
            body: Notification body
            data: Additional data payload
            
        Returns:
            True if successful
        """
        if not self.enabled:
            logger.info(f"PUSH SIMULATION: {title} - {body}")
            return False
        
        try:
            message = self.messaging.Message(
                notification=self.messaging.Notification(
                    title=title,
                    body=body,
                ),
                data=data or {},
                topic=self.topic,
            )
            
            response = self.messaging.send(message)
            logger.info(f"Push notification sent: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return False
    
    def send_alert_notification(self, alert_data: dict) -> bool:
        """
        Send formatted alert notification.
        
        Args:
            alert_data: Alert information dictionary
            
        Returns:
            True if successful
        """
        priority = alert_data.get('priority', 'UNKNOWN')
        alert_type = alert_data.get('type', 'UNKNOWN')
        camera = alert_data.get('camera_id', 'N/A')
        
        title = f"🚨 {priority} Alert"
        body = f"{alert_type} detected on Camera {camera}"
        
        data = {
            'alert_id': str(alert_data.get('id', '')),
            'priority': priority,
            'type': alert_type,
            'camera_id': str(camera),
            'timestamp': str(alert_data.get('timestamp', ''))
        }
        
        return self.send_notification(title, body, data)
