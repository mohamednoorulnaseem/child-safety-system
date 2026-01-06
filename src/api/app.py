"""
Flask Application
RESTful API for Child Safety System
"""
from flask import Flask
from flask_cors import CORS
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import API
from src.utils.logger import setup_logger
from src.api.routes import api_bp

logger = setup_logger('FlaskApp')


def create_app():
    """
    Create and configure Flask application.
    
    Returns:
        Configured Flask app instance
    """
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Health check endpoint
    @app.route('/')
    def index():
        return {
            'status': 'running',
            'service': 'Child Safety System API',
            'version': '1.0.0'
        }
    
    logger.info("Flask application created")
    
    return app


def run_api_server():
    """Run Flask API server."""
    app = create_app()
    
    logger.info(f"Starting API server on {API['host']}:{API['port']}")
    
    app.run(
        host=API['host'],
        port=API['port'],
        debug=API['debug'],
        threaded=True
    )


if __name__ == '__main__':
    run_api_server()
