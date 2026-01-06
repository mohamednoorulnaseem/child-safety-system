"""Flask API modules"""
from .app import create_app
from .database import init_database, save_alert, get_recent_alerts, get_statistics

__all__ = ['create_app', 'init_database', 'save_alert', 'get_recent_alerts', 'get_statistics']
