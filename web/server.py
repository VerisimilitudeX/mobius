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
    # server.py is in the web directory; serve files from current directory
    return send_from_directory('.', 'landing.html')

@app.route('/dashboard')
def serve_dashboard():
    # Serve the main dashboard UI
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

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
def save_config():
    """Persist dashboard configuration for the R pipeline to consume."""
    try:
        config = request.json or {}
        # Stronger validation
        required = ['condition1', 'condition2', 'condition3']
        missing = [k for k in required if not config.get(k)]
        if missing:
            return jsonify({'message': f'Missing fields: {", ".join(missing)}'}), 400

        # Save for run_pipeline_master.R (reads dashboard_config.json in web_mode)
        import json
        with open('dashboard_config.json', 'w') as f:
            json.dump({
                'condition1': config['condition1'],
                'condition2': config['condition2'],
                'condition3': config['condition3'],
                'startStep': int(config.get('startStep', 0))
            }, f, indent=2)

        return jsonify({'message': 'Configuration saved'})
    except Exception as e:
        return jsonify({'message': f'Failed to save config: {e}'}), 500

@app.route('/api/pipeline/start', methods=['POST'])
def start_pipeline():
    """Start the R pipeline and stream progress to WebSocket clients."""
    try:
        # Run the pipeline in a subprocess
        import subprocess
        import threading
        import json

        payload = request.json or {}
        # Ensure config exists on disk for web_mode
        if not os.path.exists('dashboard_config.json'):
            with open('dashboard_config.json', 'w') as f:
                json.dump(payload, f, indent=2)

        cmd = ['Rscript', 'src/original/run_pipeline_master.R', str(payload.get('startStep', 0)), '--web_mode=TRUE']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def stream_output():
            try:
                for line in proc.stdout:  # type: ignore[attr-defined]
                    socketio.emit('output', {'type': 'output', 'message': line})
            finally:
                code = proc.wait()
                status = 'complete' if code == 0 else 'error'
                socketio.emit(status, {'type': status, 'message': f'Pipeline {status}', 'code': code})

        threading.Thread(target=stream_output, daemon=True).start()
        return jsonify({'message': 'Pipeline started'})
    except Exception as e:
        return jsonify({'message': f'Failed to start pipeline: {e}'}), 500

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