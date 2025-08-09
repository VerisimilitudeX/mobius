#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup Python virtual environment
setup_venv() {
    if ! command_exists python3; then
        echo "Error: Python 3 is required but not installed."
        exit 1
    }

    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Install requirements
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Function to generate a secure secret key
generate_secret() {
    python3 -c 'import secrets; print(secrets.token_hex(32))'
}

# Parse command line arguments
MODE="local"
PORT=5000
HOST="127.0.0.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            MODE="remote"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
setup_venv

# Generate secret key if not exists
if [ ! -f ".env" ]; then
    echo "Generating secret key..."
    echo "SECRET_KEY=$(generate_secret)" > .env
    echo "Environment file created with new secret key"
fi

# Start server based on mode
if [ "$MODE" = "remote" ]; then
    echo "Starting server in remote mode..."
    echo "Server will be accessible from: $HOST:$PORT"
    python server.py --host "$HOST" --port "$PORT"
else
    echo "Starting server in local mode..."
    echo "Server will be accessible at: http://localhost:$PORT"
    python server.py
fi