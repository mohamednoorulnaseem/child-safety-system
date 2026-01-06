"""
Alert Manager Module
Coordinates all alert channels (buzzer, SMS, push, email, database)
"""
import time
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import ALERTS
from src.utils.logger import setup_logger
from .buzzer_control import BuzzerControl
from .sms_sender import SMSSender
from .push_notifier import PushNotifier

logger = setup_logger('AlertManager')


class AlertManager:
    """
    Manages multi-channel alert delivery system.
    Coordinates buzzer, SMS, push notifications, and database logging.
    """
    
    def __init__(self):
        """Initialize all alert channels."""
        logger.info("Initializing AlertManager...")
        
        # Initialize alert channels
        self.buzzer = BuzzerControl()
        self.sms_sender = SMSSender()
        self.push_notifier = PushNotifier()
        
        # Alert cooldown tracking (prevent spam)
        self.alert_cooldown = {}
        self.cooldown_period = ALERTS['alert_cooldown']
        
        logger.info("AlertManager initialized successfully")
    
    def trigger_alert(self, alert_data: Dict) -> bool:
        """
        Trigger multi-channel alert.
        
        Args:
            alert_data: Dictionary containing:
                - priority: 'HIGH'|'MEDIUM'|'LOW'
                - type: Alert type string
                - track_id: int
                - camera_id: int
                - timestamp: str or float
                - confidence: float
                - details: dict
                
        Returns:
            True if alert was triggered (not in cooldown)
        """
        priority = alert_data.get('priority', 'MEDIUM')
        alert_type = alert_data.get('type', 'UNKNOWN')
        track_id = alert_data.get('track_id')
        
        # Check cooldown
        if self._is_in_cooldown(track_id, alert_type):
            logger.debug(f"Alert in cooldown: {alert_type} for track {track_id}")
            return False
        
        logger.info(f"Triggering {priority} alert: {alert_type}")
        
        # Activate channels based on priority
        if priority == 'HIGH':
            self._activate_all_channels(alert_data)
        elif priority == 'MEDIUM':
            self._activate_medium_channels(alert_data)
        elif priority == 'LOW':
            self._activate_low_channels(alert_data)
        
        # Update cooldown
        self._update_cooldown(track_id, alert_type)
        
        # Log to database (if database module is available)
        self._log_to_database(alert_data)
        
        return True
    
    def _activate_all_channels(self, alert_data: Dict):
        """Activate all alert channels (HIGH priority)."""
        # Physical buzzer - urgent pattern
        self.buzzer.activate(duration=ALERTS['buzzer_duration'], pattern='urgent')
        
        # SMS
        self.sms_sender.send_alert_sms(alert_data)
        
        # Push notification
        self.push_notifier.send_alert_notification(alert_data)
        
        logger.info("All alert channels activated")
    
    def _activate_medium_channels(self, alert_data: Dict):
        """Activate medium priority channels."""
        # Physical buzzer - beep pattern
        self.buzzer.activate(duration=1.0, pattern='beep')
        
        # Push notification only
        self.push_notifier.send_alert_notification(alert_data)
        
        logger.info("Medium priority channels activated")
    
    def _activate_low_channels(self, alert_data: Dict):
        """Activate low priority channels (log only)."""
        logger.info("Low priority alert - logging only")
    
    def _is_in_cooldown(self, track_id: int, alert_type: str) -> bool:
        """
        Check if alert is in cooldown period.
        
        Args:
            track_id: Track ID
            alert_type: Alert type
            
        Returns:
            True if in cooldown, False otherwise
        """
        key = f"{track_id}_{alert_type}"
        
        if key not in self.alert_cooldown:
            return False
        
        last_alert_time = self.alert_cooldown[key]
        elapsed = time.time() - last_alert_time
        
        return elapsed < self.cooldown_period
    
    def _update_cooldown(self, track_id: int, alert_type: str):
        """Update cooldown timestamp for alert."""
        key = f"{track_id}_{alert_type}"
        self.alert_cooldown[key] = time.time()
    
    def _log_to_database(self, alert_data: Dict):
        """
        Log alert to database.
        
        Args:
            alert_data: Alert information
        """
        try:
            # Import database module
            from src.api.database import save_alert
            
            # Save to database
            save_alert(alert_data)
            logger.debug("Alert logged to database")
            
        except ImportError:
            logger.debug("Database module not available - skipping log")
        except Exception as e:
            logger.error(f"Failed to log alert to database: {e}")
    
    def create_alert_data(self, alert_type: str, track_id: int, 
                         camera_id: int, confidence: float,
                         details: Dict = None) -> Dict:
        """
        Create standardized alert data dictionary.
        
        Args:
            alert_type: Type of alert
            track_id: Track ID
            camera_id: Camera ID
            confidence: Confidence score
            details: Additional details
            
        Returns:
            Alert data dictionary
        """
        # Determine priority based on confidence and type
        priority = self._determine_priority(alert_type, confidence, details)
        
        alert_data = {
            'priority': priority,
            'type': alert_type,
            'track_id': track_id,
            'camera_id': camera_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confidence': confidence,
            'details': details or {}
        }
        
        return alert_data
    
    def _determine_priority(self, alert_type: str, confidence: float,
                           details: Dict = None) -> str:
        """
        Determine alert priority based on type, confidence, and details.
        
        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        # HIGH priority conditions
        if confidence >= ALERTS['high_priority_threshold']:
            return 'HIGH'
        
        # Check for multiple indicators
        if details:
            indicator_count = sum([
                details.get('distressed', False),
                details.get('struggling', False),
                details.get('being_dragged', False),
                details.get('suspicious_adult_nearby', False)
            ])
            
            if indicator_count >= 2:
                return 'HIGH'
        
        # MEDIUM priority
        if confidence >= ALERTS['suspicious_confidence']:
            return 'MEDIUM'
        
        # LOW priority
        return 'LOW'
    
    def cleanup(self):
        """Cleanup alert resources."""
        self.buzzer.cleanup()
        logger.info("AlertManager cleanup complete")


# Test code
if __name__ == '__main__':
    print("Testing AlertManager...")
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Test different priority alerts
    test_alerts = [
        {
            'type': 'CHILD_DISTRESS',
            'track_id': 1,
            'camera_id': 1,
            'confidence': 0.95,
            'details': {'distressed': True, 'struggling': True}
        },
        {
            'type': 'UNATTENDED_CHILD',
            'track_id': 2,
            'camera_id': 2,
            'confidence': 0.78,
            'details': {'duration': 600}
        },
        {
            'type': 'SUSPICIOUS_BEHAVIOR',
            'track_id': 3,
            'camera_id': 1,
            'confidence': 0.65,
            'details': {}
        }
    ]
    
    for i, alert_info in enumerate(test_alerts, 1):
        print(f"\n--- Test Alert {i} ---")
        
        alert_data = alert_manager.create_alert_data(**alert_info)
        
        print(f"Priority: {alert_data['priority']}")
        print(f"Type: {alert_data['type']}")
        print(f"Confidence: {alert_data['confidence']}")
        
        success = alert_manager.trigger_alert(alert_data)
        
        if success:
            print("Alert triggered successfully")
        else:
            print("Alert in cooldown")
        
        time.sleep(2)
    
    # Cleanup
    alert_manager.cleanup()
    
    print("\nAlert system test complete")
