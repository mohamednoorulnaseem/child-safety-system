"""
Database Module
SQLite database operations for alerts and system logs
"""
import sqlite3
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import DATABASE
from src.utils.logger import setup_logger

logger = setup_logger('Database')


def get_connection():
    """Get database connection."""
    db_path = DATABASE['path']
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_database():
    """
    Initialize database with required tables.
    """
    logger.info("Initializing database...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                priority TEXT NOT NULL,
                type TEXT NOT NULL,
                camera_id INTEGER NOT NULL,
                track_id INTEGER,
                confidence REAL,
                details TEXT,
                acknowledged BOOLEAN DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT
            )
        ''')
        
        # Trusted persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trusted_persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                relationship TEXT,
                face_embedding BLOB,
                photo_path TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # System logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT,
                module TEXT,
                message TEXT
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        conn.rollback()
    finally:
        conn.close()


def save_alert(alert_data: Dict) -> Optional[int]:
    """
    Save alert to database.
    
    Args:
        alert_data: Alert information dictionary
        
    Returns:
        Alert ID if successful, None otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO alerts (
                timestamp, priority, type, camera_id, track_id, 
                confidence, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            alert_data['priority'],
            alert_data['type'],
            alert_data['camera_id'],
            alert_data.get('track_id'),
            alert_data.get('confidence'),
            json.dumps(alert_data.get('details', {}))
        ))
        
        conn.commit()
        alert_id = cursor.lastrowid
        logger.info(f"Alert saved to database: ID {alert_id}")
        return alert_id
        
    except Exception as e:
        logger.error(f"Failed to save alert: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def get_recent_alerts(hours: int = 24, limit: int = 100) -> List[Dict]:
    """
    Get recent alerts from database.
    
    Args:
        hours: Number of hours to look back
        limit: Maximum number of alerts to return
        
    Returns:
        List of alert dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cutoff_time = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute('''
            SELECT id, timestamp, priority, type, camera_id, track_id,
                   confidence, details, acknowledged, acknowledged_by, acknowledged_at
            FROM alerts
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (cutoff_time, limit))
        
        rows = cursor.fetchall()
        
        alerts = []
        for row in rows:
            alert = {
                'id': row[0],
                'timestamp': row[1],
                'priority': row[2],
                'type': row[3],
                'camera_id': row[4],
                'track_id': row[5],
                'confidence': row[6],
                'details': json.loads(row[7]) if row[7] else {},
                'acknowledged': bool(row[8]),
                'acknowledged_by': row[9],
                'acknowledged_at': row[10]
            }
            alerts.append(alert)
        
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        return []
    finally:
        conn.close()


def acknowledge_alert(alert_id: int, acknowledged_by: str) -> bool:
    """
    Mark alert as acknowledged.
    
    Args:
        alert_id: Alert ID
        acknowledged_by: Name/ID of person acknowledging
        
    Returns:
        True if successful
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE alerts
            SET acknowledged = 1,
                acknowledged_by = ?,
                acknowledged_at = ?
            WHERE id = ?
        ''', (acknowledged_by, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), alert_id))
        
        conn.commit()
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_statistics() -> Dict:
    """
    Get alert statistics.
    
    Returns:
        Dictionary with statistics
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Total alerts today
        cursor.execute(
            "SELECT COUNT(*) FROM alerts WHERE timestamp >= ?",
            (today,)
        )
        today_total = cursor.fetchone()[0]
        
        # By priority
        cursor.execute('''
            SELECT priority, COUNT(*) 
            FROM alerts 
            WHERE timestamp >= ?
            GROUP BY priority
        ''', (today,))
        by_priority = dict(cursor.fetchall())
        
        # By type
        cursor.execute('''
            SELECT type, COUNT(*) 
            FROM alerts 
            WHERE timestamp >= ?
            GROUP BY type
        ''', (today,))
        by_type = dict(cursor.fetchall())
        
        # By camera
        cursor.execute('''
            SELECT camera_id, COUNT(*) 
            FROM alerts 
            WHERE timestamp >= ?
            GROUP BY camera_id
        ''', (today,))
        by_camera = {f"camera_{k}": v for k, v in cursor.fetchall()}
        
        stats = {
            'today_total': today_total,
            'by_priority': by_priority,
            'by_type': by_type,
            'by_camera': by_camera
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {}
    finally:
        conn.close()


def save_system_log(level: str, module: str, message: str):
    """
    Save system log entry.
    
    Args:
        level: Log level (INFO, WARNING, ERROR)
        module: Module name
        message: Log message
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO system_logs (timestamp, level, module, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), level, module, message))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Failed to save system log: {e}")
        conn.rollback()
    finally:
        conn.close()


# Initialize database on import
if __name__ != '__main__':
    try:
        init_database()
    except Exception as e:
        logger.error(f"Failed to auto-initialize database: {e}")
