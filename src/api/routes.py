"""
API Routes
RESTful endpoints for Child Safety System
"""
from flask import Blueprint, jsonify, request
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger
from .database import (
    get_recent_alerts, get_statistics, acknowledge_alert, save_alert
)

logger = setup_logger('API')

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/alerts/recent', methods=['GET'])
def get_alerts():
    """
    Get recent alerts.
    Query params: hours (default 24), limit (default 100)
    """
    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 100))
        
        alerts = get_recent_alerts(hours, limit)
        
        return jsonify({
            'success': True,
            'count': len(alerts),
            'alerts': alerts
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/alerts/stats', methods=['GET'])
def get_stats():
    """
    Get alert statistics.
    """
    try:
        stats = get_statistics()
        
        return jsonify({
            'success': True,
            **stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/alerts', methods=['POST'])
def create_alert():
    """
    Create new alert (called by detection system).
    """
    try:
        alert_data = request.get_json()
        
        if not alert_data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        alert_id = save_alert(alert_data)
        
        if alert_id:
            return jsonify({
                'success': True,
                'alert_id': alert_id
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save alert'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def ack_alert(alert_id):
    """
    Acknowledge an alert.
    """
    try:
        data = request.get_json()
        acknowledged_by = data.get('acknowledged_by', 'Unknown')
        
        success = acknowledge_alert(alert_id, acknowledged_by)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Alert acknowledged'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to acknowledge alert'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@api_bp.route('/cameras/status', methods=['GET'])
def camera_status():
    """
    Get camera status (placeholder - implement with actual camera monitoring).
    """
    # This would be connected to actual camera monitoring
    return jsonify({
        'camera_1': {'status': 'active', 'fps': 30, 'last_detection': 'Just now'},
        'camera_2': {'status': 'active', 'fps': 28, 'last_detection': '5 sec ago'},
        'camera_3': {'status': 'active', 'fps': 30, 'last_detection': '2 sec ago'}
    }), 200


@api_bp.route('/system/health', methods=['GET'])
def system_health():
    """
    Get system health status.
    """
    import psutil
    
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine status
        if cpu_usage > 90 or memory.percent > 90:
            status = 'degraded'
        elif cpu_usage > 95 or memory.percent > 95:
            status = 'down'
        else:
            status = 'healthy'
        
        return jsonify({
            'status': status,
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'uptime': psutil.boot_time()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return jsonify({
            'status': 'unknown',
            'error': str(e)
        }), 500
