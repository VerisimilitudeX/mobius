class ConnectionManager {
    constructor() {
        this.serverUrl = '';
        this.socket = null;
        this.isConnected = false;
        this.authToken = localStorage.getItem('authToken');
        this.connectionType = localStorage.getItem('connectionType') || 'local';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.listeners = new Set();
        
        // Initialize connection indicator
        this.indicator = document.createElement('div');
        this.indicator.className = 'connection-indicator';
        this.indicator.textContent = 'Disconnected';
        document.querySelector('.nav-list').appendChild(this.indicator);
        
        this.initializeUI();
    }

    initializeUI() {
        // Add connection type switcher
        const switcher = document.createElement('div');
        switcher.className = 'connection-switcher';
        switcher.innerHTML = `
            <select id="connectionType" class="connection-select">
                <option value="local">Local Connection</option>
                <option value="remote">Remote Connection</option>
            </select>
        `;
        document.querySelector('.nav-list').insertBefore(switcher, this.indicator);

        const select = switcher.querySelector('select');
        select.value = this.connectionType;
        select.addEventListener('change', (e) => this.handleConnectionTypeChange(e.target.value));

        // Initialize connection modal
        this.createConnectionModal();
    }

    createConnectionModal() {
        const modal = document.createElement('div');
        modal.className = 'auth-modal hidden';
        modal.innerHTML = `
            <h3>Connect to Remote Server</h3>
            <form id="connectionForm">
                <div class="form-group">
                    <label for="serverUrl">Server URL</label>
                    <input type="url" id="serverUrl" required placeholder="https://your-server.com">
                </div>
                <div class="form-group">
                    <label for="apiKey">API Key</label>
                    <input type="password" id="apiKey" required>
                </div>
                <button type="submit" class="btn btn-primary w-full">Connect</button>
            </form>
        `;
        document.body.appendChild(modal);

        const form = modal.querySelector('form');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.connectToRemote(
                form.querySelector('#serverUrl').value,
                form.querySelector('#apiKey').value
            );
        });
    }

    async connectToRemote(url, apiKey) {
        try {
            const response = await fetch(`${url}/api/auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ apiKey })
            });

            if (!response.ok) {
                throw new Error('Authentication failed');
            }

            const { token } = await response.json();
            this.authToken = token;
            localStorage.setItem('authToken', token);
            this.serverUrl = url;
            localStorage.setItem('serverUrl', url);

            this.setupWebSocket();
            document.querySelector('.auth-modal').classList.add('hidden');
            showNotification('Connected', 'Successfully connected to remote server', 'success');
        } catch (error) {
            showNotification('Connection Failed', error.message, 'error');
        }
    }

    setupWebSocket() {
        if (this.socket) {
            this.socket.close();
        }

        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = this.connectionType === 'local' 
            ? `${wsProtocol}//${window.location.host}/ws`
            : `${wsProtocol}//${new URL(this.serverUrl).host}/ws`;

        this.socket = new WebSocket(wsUrl);
        
        this.socket.onopen = () => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus('connected');
            this.notifyListeners('connected');

            // Send authentication if remote
            if (this.connectionType === 'remote' && this.authToken) {
                this.socket.send(JSON.stringify({
                    type: 'auth',
                    token: this.authToken
                }));
            }
        };

        this.socket.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            this.notifyListeners('disconnected');

            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
                setTimeout(() => this.setupWebSocket(), delay);
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.notifyListeners('error', error);
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.notifyListeners('message', data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
    }

    updateConnectionStatus(status) {
        this.indicator.className = `connection-indicator ${status}`;
        this.indicator.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
    }

    handleConnectionTypeChange(type) {
        this.connectionType = type;
        localStorage.setItem('connectionType', type);

        if (type === 'remote' && !this.authToken) {
            document.querySelector('.auth-modal').classList.remove('hidden');
        } else {
            this.setupWebSocket();
        }
    }

    addListener(callback) {
        this.listeners.add(callback);
    }

    removeListener(callback) {
        this.listeners.delete(callback);
    }

    notifyListeners(event, data) {
        this.listeners.forEach(callback => callback(event, data));
    }

    sendMessage(message) {
        if (this.isConnected) {
            this.socket.send(JSON.stringify(message));
        } else {
            console.error('Cannot send message: Not connected');
            showNotification('Error', 'Not connected to server', 'error');
        }
    }
}

// Create and export connection manager instance
const connectionManager = new ConnectionManager();
export default connectionManager;