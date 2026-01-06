"""
Unit tests for alert system
"""
import pytest
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.alerts import AlertManager, BuzzerControl, SMSSender, PushNotifier


@pytest.fixture
def alert_manager():
    """Create AlertManager instance."""
    return AlertManager()


@pytest.fixture
def buzzer_control():
    """Create BuzzerControl instance."""
    return BuzzerControl()


@pytest.fixture
def sms_sender():
    """Create SMSSender instance."""
    return SMSSender()


@pytest.fixture
def push_notifier():
    """Create PushNotifier instance."""
    return PushNotifier()


@pytest.fixture
def sample_alert_data():
    """Create sample alert data."""
    return {
        'priority': 'HIGH',
        'type': 'CHILD_DISTRESS',
        'track_id': 1,
        'camera_id': 1,
        'timestamp': '2026-01-06 10:30:00',
        'confidence': 0.95,
        'details': {
            'distressed': True,
            'struggling': True
        }
    }


def test_alert_manager_initialization(alert_manager):
    """Test AlertManager initializes correctly."""
    assert alert_manager is not None
    assert alert_manager.buzzer is not None
    assert alert_manager.sms_sender is not None
    assert alert_manager.push_notifier is not None


def test_buzzer_activation(buzzer_control):
    """Test buzzer activation (simulation)."""
    # Should not raise error even if GPIO not available
    try:
        buzzer_control.activate(duration=0.1, pattern='continuous')
        assert True
    except Exception as e:
        pytest.fail(f"Buzzer activation failed: {e}")


def test_sms_sending(sms_sender):
    """Test SMS sending (simulation if credentials not configured)."""
    result = sms_sender.send_sms("Test message")
    # Should return False if not configured, but not crash
    assert isinstance(result, bool)


def test_push_notification(push_notifier):
    """Test push notification (simulation if Firebase not configured)."""
    result = push_notifier.send_notification("Test", "Test message")
    # Should return False if not configured, but not crash
    assert isinstance(result, bool)


def test_alert_manager_create_alert_data(alert_manager):
    """Test alert data creation."""
    alert_data = alert_manager.create_alert_data(
        alert_type='CHILD_DISTRESS',
        track_id=1,
        camera_id=1,
        confidence=0.95,
        details={'distressed': True}
    )
    
    assert 'priority' in alert_data
    assert 'type' in alert_data
    assert 'track_id' in alert_data
    assert 'camera_id' in alert_data
    assert 'timestamp' in alert_data
    assert 'confidence' in alert_data
    assert 'details' in alert_data


def test_alert_priority_determination(alert_manager):
    """Test alert priority determination logic."""
    # High confidence -> HIGH priority
    priority = alert_manager._determine_priority(
        'CHILD_DISTRESS', 0.95, {'distressed': True}
    )
    assert priority == 'HIGH'
    
    # Medium confidence -> MEDIUM priority
    priority = alert_manager._determine_priority(
        'UNATTENDED_CHILD', 0.75, {}
    )
    assert priority == 'MEDIUM'
    
    # Low confidence -> LOW priority
    priority = alert_manager._determine_priority(
        'SUSPICIOUS_BEHAVIOR', 0.65, {}
    )
    assert priority == 'LOW'


def test_alert_cooldown(alert_manager, sample_alert_data):
    """Test alert cooldown prevents spam."""
    # First alert should trigger
    result1 = alert_manager.trigger_alert(sample_alert_data)
    assert result1 == True
    
    # Immediate second alert should be in cooldown
    result2 = alert_manager.trigger_alert(sample_alert_data)
    assert result2 == False
    
    # Different track should trigger
    sample_alert_data['track_id'] = 2
    result3 = alert_manager.trigger_alert(sample_alert_data)
    assert result3 == True


def test_alert_manager_cleanup(alert_manager):
    """Test alert manager cleanup."""
    try:
        alert_manager.cleanup()
        assert True
    except Exception as e:
        pytest.fail(f"Cleanup failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
