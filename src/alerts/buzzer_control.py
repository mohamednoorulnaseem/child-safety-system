"""
Buzzer Control Module
Controls GPIO buzzer for physical alerts on Raspberry Pi
"""
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import GPIO_PINS, ENABLE_GPIO
from src.utils.logger import setup_logger

logger = setup_logger('BuzzerControl')


class BuzzerControl:
    """
    Controls GPIO buzzer for physical alerts.
    """
    
    def __init__(self):
        """Initialize buzzer control."""
        self.buzzer_pin = GPIO_PINS['buzzer']
        self.gpio_enabled = ENABLE_GPIO
        
        if self.gpio_enabled:
            try:
                import RPi.GPIO as GPIO
                self.GPIO = GPIO
                
                # Setup GPIO
                self.GPIO.setmode(GPIO.BCM)
                self.GPIO.setup(self.buzzer_pin, GPIO.OUT)
                self.GPIO.output(self.buzzer_pin, GPIO.LOW)
                
                logger.info(f"Buzzer initialized on GPIO pin {self.buzzer_pin}")
            except ImportError:
                logger.warning("RPi.GPIO not available - buzzer disabled")
                self.gpio_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize GPIO: {e}")
                self.gpio_enabled = False
        else:
            logger.info("GPIO disabled in config - buzzer will simulate only")
    
    def activate(self, duration: float = 2.0, pattern: str = 'continuous'):
        """
        Activate buzzer with specified pattern.
        
        Args:
            duration: Duration in seconds
            pattern: 'continuous', 'beep', or 'urgent'
        """
        if not self.gpio_enabled:
            logger.info(f"BUZZER SIMULATION: {pattern} for {duration}s")
            return
        
        try:
            if pattern == 'continuous':
                self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
                time.sleep(duration)
                self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
            
            elif pattern == 'beep':
                end_time = time.time() + duration
                while time.time() < end_time:
                    self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
                    time.sleep(0.2)
                    self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
                    time.sleep(0.2)
            
            elif pattern == 'urgent':
                end_time = time.time() + duration
                while time.time() < end_time:
                    self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
                    time.sleep(0.1)
                    self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)
                    time.sleep(0.1)
            
            logger.info(f"Buzzer activated: {pattern} for {duration}s")
            
        except Exception as e:
            logger.error(f"Buzzer activation failed: {e}")
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self.gpio_enabled:
            try:
                self.GPIO.cleanup(self.buzzer_pin)
                logger.info("Buzzer GPIO cleaned up")
            except Exception as e:
                logger.error(f"GPIO cleanup failed: {e}")
    
    def __del__(self):
        """Destructor - cleanup on object deletion."""
        self.cleanup()
