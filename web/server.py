from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import jwt
import os
import secrets
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['JWT_EXPIRATION_HOURS'] = 24

# Store active connections and their tokens
active_connections = {}
api_keys = set()  # In production, this should be in a secure database

def generate_token():
    expiration = datetime.utcnow() + timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
    return jwt.encode(
        {'exp': expiration},
        app.config['SECRET_KEY'],
        algorithm='HS256'
    )

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except:
            return jsonify({'message': 'Token is invalid'}), 401

        return f(*args, **kwargs)
    return decorated

# Serve static files
@app.route('/')
def serve_landing():
    return send_from_directory('web', 'landing.html')

@app.route('/dashboard')
def serve_dashboard():
    return send_from_directory('web', 'dashboard.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

# API Routes
@app.route('/api/auth', methods=['POST'])
def authenticate():
    data = request.json
    api_key = data.get('apiKey')
    
    if not api_key or api_key not in api_keys:
        return jsonify({'message': 'Invalid API key'}), 401
    
    token = generate_token()
    return jsonify({'token': token})

@app.route('/api/config', methods=['POST'])
@token_required
def save_config():
    config = request.json
    # Save configuration logic here
    return jsonify({'message': 'Configuration saved'})

@app.route('/api/pipeline/start', methods=['POST'])
@token_required
def start_pipeline():
    config = request.json
    # Start pipeline logic here
    return jsonify({'message': 'Pipeline started'})

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    if request.sid in active_connections:
        del active_connections[request.sid]

@socketio.on('auth')
def handle_authentication(data):
    try:
        token = data.get('token')
        if token:
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            active_connections[request.sid] = {'authenticated': True}
            emit('auth_success')
        else:
            emit('auth_failed', {'message': 'No token provided'})
    except jwt.ExpiredSignatureError:
        emit('auth_failed', {'message': 'Token expired'})
    except jwt.InvalidTokenError:
        emit('auth_failed', {'message': 'Invalid token'})

@socketio.on('pipeline_command')
def handle_pipeline_command(data):
    if not active_connections.get(request.sid, {}).get('authenticated'):
        emit('error', {'message': 'Not authenticated'})
        return
    
    command = data.get('command')
    # Handle pipeline commands here
    emit('command_response', {'status': 'success'})

def create_api_key():
    """Generate a new API key and add it to the set of valid keys"""
    api_key = secrets.token_urlsafe(32)
    api_keys.add(api_key)
    return api_key

if __name__ == '__main__':
    # Generate initial API key if none exist
    if not api_keys:
        initial_key = create_api_key()
        print(f"Initial API Key: {initial_key}")
    
    socketio.run(app, debug=True)