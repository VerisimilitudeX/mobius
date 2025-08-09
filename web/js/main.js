import connectionManager from './connection.js';

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const configForm = document.getElementById('configForm');
    const startPipelineBtn = document.getElementById('startPipeline');
    const progressBar = document.querySelector('.progress');
    const progressText = document.getElementById('progressText');
    const currentStep = document.getElementById('currentStep');
    const elapsedTime = document.getElementById('elapsedTime');
    const outputLog = document.querySelector('#outputLog pre code');
    
    // Pipeline state
    let pipelineRunning = false;
    let startTime = null;
    let elapsedTimer = null;
    let currentStepNumber = 0;
    const totalSteps = 16;

    // WebSocket state
    let ws = null;
    let wsReconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    const reconnectDelay = 1000; // Start with 1 second

    // Input validation state
    const requiredInputs = document.querySelectorAll('input[required], select[required]');
    let formValid = false;

    function setupWebSocket() {
        if (ws) {
            ws.close();
        }

        ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onopen = () => {
            wsReconnectAttempts = 0;
            showNotification('Connected', 'WebSocket connection established', 'success');
            updateConnectionStatus('connected');
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
                appendToLog('Error: Failed to parse server message');
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus('error');
            appendToLog('Error: WebSocket connection error. Some real-time updates may not be available.');
        };

        ws.onclose = () => {
            updateConnectionStatus('disconnected');
            if (wsReconnectAttempts < maxReconnectAttempts) {
                wsReconnectAttempts++;
                const delay = Math.min(reconnectDelay * Math.pow(2, wsReconnectAttempts - 1), 30000);
                showNotification('Disconnected', `Reconnecting in ${delay/1000} seconds...`, 'warning');
                setTimeout(setupWebSocket, delay);
            } else {
                showNotification('Connection Failed', 'Unable to establish WebSocket connection', 'error');
            }
        };
    }

    function updateConnectionStatus(status) {
        const statusIndicator = document.getElementById('connectionStatus');
        if (statusIndicator) {
            statusIndicator.className = `connection-status ${status}`;
            statusIndicator.setAttribute('title', `Connection: ${status}`);
        }
    }

    // Form validation
    function validateForm() {
        formValid = true;
        let firstError = null;

        requiredInputs.forEach(input => {
            const value = input.value.trim();
            const isValid = validateInput(input, value);
            
            if (!isValid && !firstError) {
                firstError = input;
            }
            
            formValid = formValid && isValid;
        });

        startPipelineBtn.disabled = !formValid;

        if (firstError) {
            firstError.focus();
        }

        return formValid;
    }

    function validateInput(input, value) {
        const isEmpty = !value;
        const parent = input.closest('.form-group');
        
        if (isEmpty) {
            showInputError(input, 'This field is required');
            parent?.classList.add('has-error');
            return false;
        }

        // Path validation for condition inputs
        if (input.id.startsWith('condition')) {
            if (!isValidPath(value)) {
                showInputError(input, 'Please enter a valid directory path');
                parent?.classList.add('has-error');
                return false;
            }
        }

        hideInputError(input);
        parent?.classList.remove('has-error');
        return true;
    }

    function isValidPath(path) {
        // Basic path validation - can be expanded based on requirements
        return path.length > 0 && !path.includes('*') && !path.includes('|');
    }

    requiredInputs.forEach(input => {
        input.addEventListener('input', () => {
            validateInput(input, input.value.trim());
            validateForm();
        });

        input.addEventListener('blur', () => {
            validateInput(input, input.value.trim());
        });

        input.addEventListener('focus', () => {
            hideInputError(input);
        });
    });

    function showInputError(input, message) {
        let errorDiv = input.parentElement.querySelector('.input-error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'input-error';
            input.parentElement.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
        errorDiv.style.opacity = '1';
    }

    function hideInputError(input) {
        const errorDiv = input.parentElement.querySelector('.input-error');
        if (errorDiv) {
            errorDiv.style.opacity = '0';
            setTimeout(() => errorDiv.remove(), 300);
        }
    }

    // Form submission
    configForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!pipelineRunning && validateForm()) {
            startPipeline();
        }
    });

    startPipelineBtn.addEventListener('click', () => {
        if (!pipelineRunning && validateForm()) {
            configForm.requestSubmit();
        }
    });

    // Connect to pipeline server
    connectionManager.addListener((event, data) => {
        switch (event) {
            case 'connected':
                showNotification('Connected', 'Successfully connected to pipeline server', 'success');
                break;
            case 'disconnected':
                showNotification('Disconnected', 'Lost connection to pipeline server', 'warning');
                break;
            case 'message':
                handleWebSocketMessage(data);
                break;
            case 'error':
                showNotification('Error', 'Connection error occurred', 'error');
                break;
        }
    });

    // Pipeline control
    async function startPipeline() {
        const config = {
            condition1: document.getElementById('condition1').value,
            condition2: document.getElementById('condition2').value,
            condition3: document.getElementById('condition3').value,
            startStep: document.getElementById('startStep').value
        };

        try {
            const headers = {
                'Content-Type': 'application/json'
            };

            // Add authentication token for remote connections
            if (connectionManager.connectionType === 'remote' && connectionManager.authToken) {
                headers['Authorization'] = `Bearer ${connectionManager.authToken}`;
            }

            // Save configuration
            const configResponse = await fetch('/api/config', {
                method: 'POST',
                headers,
                body: JSON.stringify(config)
            });

            if (!configResponse.ok) {
                throw new Error('Failed to save configuration');
            }

            // Start pipeline
            const startResponse = await fetch('/api/pipeline/start', {
                method: 'POST',
                headers,
                body: JSON.stringify(config)
            });

            if (!startResponse.ok) {
                throw new Error('Failed to start pipeline');
            }

            // Update UI state
            pipelineRunning = true;
            startPipelineBtn.textContent = 'Running...';
            startPipelineBtn.disabled = true;
            startTime = Date.now();
            currentStepNumber = parseInt(config.startStep);
            
            // Start elapsed time counter
            startElapsedTimer();
            
            // Clear previous output
            clearOutput();
            
            // Reset visualizations
            resetVisualizations();
            
            appendToLog('Pipeline started successfully');
            showNotification('Pipeline Started', 'Analysis is now running', 'success');

        } catch (error) {
            appendToLog(`Error: ${error.message}`);
            showNotification('Error', error.message, 'error');
            resetPipelineState();
        }
    }

    function clearOutput() {
        outputLog.innerHTML = '';
    }

    function resetVisualizations() {
        document.querySelectorAll('.visualization-container').forEach(container => {
            const img = container.querySelector('img');
            const placeholder = container.querySelector('.placeholder');
            
            if (img) img.classList.add('hidden');
            if (placeholder) placeholder.classList.remove('hidden');
            
            container.classList.remove('loading', 'loaded', 'error');
        });
    }

    // WebSocket message handler
    function handleWebSocketMessage(data) {
        try {
            switch (data.type) {
                case 'step':
                    updateStep(data.step, data.message);
                    break;
                case 'progress':
                    updateProgress(data.current, data.total);
                    break;
                case 'output':
                    appendToLog(data.message);
                    break;
                case 'result':
                    updateResults(data.results);
                    break;
                case 'visualization':
                    updateVisualization(data.visualizationType, data.path);
                    break;
                case 'complete':
                    pipelineComplete();
                    break;
                case 'error':
                    handleError(data.message);
                    break;
                case 'summary':
                    updateSummaryStats(data.summary);
                    break;
                case 'findings':
                    updateKeyFindings(data.findings);
                    break;
                case 'download':
                    handleDownloadProgress(data);
                    break;
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
            appendToLog(`Error processing server message: ${error.message}`);
        }
    }

    function updateSummaryStats(summary) {
        const animateValue = (elementId, value, suffix = '') => {
            const element = document.getElementById(elementId);
            if (!element) return;
            
            let startValue = parseInt(element.textContent) || 0;
            const endValue = parseInt(value);
            const duration = 1000;
            const steps = 60;
            const stepValue = (endValue - startValue) / steps;
            
            let currentStep = 0;
            
            const animation = setInterval(() => {
                currentStep++;
                const currentValue = Math.floor(startValue + (stepValue * currentStep));
                element.textContent = currentValue + suffix;
                
                if (currentStep >= steps) {
                    element.textContent = value + suffix;
                    clearInterval(animation);
                }
            }, duration / steps);
        };
        
        animateValue('totalSamples', summary.totalSamples);
        animateValue('featuresAnalyzed', summary.featuresAnalyzed);
        animateValue('overallQuality', summary.qualityScore, '%');
    }

    function updateKeyFindings(findings) {
        const findingsContainer = document.getElementById('keyFindings');
        findingsContainer.innerHTML = '';
        
        findings.forEach(finding => {
            const findingElement = document.createElement('div');
            findingElement.className = `finding-item ${finding.type || ''}`;
            findingElement.innerHTML = `
                <p>${finding.message}</p>
                ${finding.value ? `<strong>${finding.value}</strong>` : ''}
            `;
            findingsContainer.appendChild(findingElement);
        });
    }

    function handleDownloadProgress(data) {
        const button = document.querySelector(`button[data-file="${data.filename}"]`);
        if (!button) return;
        
        if (data.status === 'preparing') {
            button.disabled = true;
            button.innerHTML = '<span>Preparing...</span>';
        } else if (data.status === 'ready') {
            button.disabled = false;
            button.innerHTML = '<span>Download</span>';
        } else if (data.status === 'error') {
            button.disabled = false;
            button.innerHTML = '<span>Retry</span>';
            showNotification('Download Error', data.message, 'error');
        }
    }

    // UI update functions
    function updateStep(step, message) {
        currentStepNumber = step;
        
        const stepElement = document.getElementById('currentStep');
        stepElement.textContent = message;
        stepElement.classList.add('status-update');
        
        // Add status class based on message content
        if (message.toLowerCase().includes('error')) {
            stepElement.classList.add('status-error');
        } else if (message.toLowerCase().includes('complete')) {
            stepElement.classList.add('status-success');
        } else {
            stepElement.classList.add('status-running');
        }
        
        setTimeout(() => {
            stepElement.classList.remove('status-update');
        }, 300);
        
        updateProgress(step, totalSteps);
    }

    function updateProgress(current, total) {
        const percentage = (current / total) * 100;
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `${current}/${total} steps completed`;
        
        // Update progress bar color based on completion
        progressBar.classList.remove('complete', 'almost-complete', 'in-progress');
        if (percentage === 100) {
            progressBar.classList.add('complete');
        } else if (percentage >= 66) {
            progressBar.classList.add('almost-complete');
        } else {
            progressBar.classList.add('in-progress');
        }
    }

    function appendToLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        const formattedMessage = `[${timestamp}] ${message}`;
        
        const logLine = document.createElement('div');
        logLine.className = 'log-line';
        
        // Add color coding based on message type
        if (message.toLowerCase().includes('error')) {
            logLine.classList.add('error-message');
        } else if (message.toLowerCase().includes('warning')) {
            logLine.classList.add('warning-message');
        } else if (message.toLowerCase().includes('success') || 
                   message.toLowerCase().includes('complete')) {
            logLine.classList.add('success-message');
        }
        
        logLine.textContent = formattedMessage;
        outputLog.appendChild(logLine);
        
        // Auto-scroll to bottom
        const container = outputLog.parentElement;
        container.scrollTop = container.scrollHeight;
        
        // Limit log size
        while (outputLog.childNodes.length > 1000) {
            outputLog.removeChild(outputLog.firstChild);
        }
    }

    function updateResults(results) {
        // Update metrics with animation
        function animateValue(element, value) {
            if (!element) return;
            
            element.classList.add('updating');
            
            // Format value based on type
            let displayValue = value;
            if (typeof value === 'number') {
                displayValue = value.toFixed(4);
            }
            
            element.textContent = displayValue;
            
            setTimeout(() => {
                element.classList.remove('updating');
            }, 300);
        }

        try {
            if (results.qualityMetrics) {
                animateValue(document.getElementById('qualityScore'), 
                           results.qualityMetrics.score);
                animateValue(document.getElementById('detectionPvalue'), 
                           results.qualityMetrics.pvalue);
            }
            
            if (results.classification) {
                animateValue(document.getElementById('accuracy'), 
                           results.classification.accuracy);
                animateValue(document.getElementById('f1Score'), 
                           results.classification.f1);
            }
            
            if (results.features) {
                animateValue(document.getElementById('selectedFeatures'), 
                           results.features.selected);
                animateValue(document.getElementById('significantDMPs'), 
                           results.features.significant);
            }
        } catch (error) {
            console.error('Error updating results:', error);
            appendToLog(`Error updating results: ${error.message}`);
        }
    }

    function updateVisualization(type, path) {
        const container = document.getElementById(`${type}Plot`);
        if (!container) return;

        const img = container.querySelector('img');
        const placeholder = container.querySelector('.placeholder');
        
        // Add loading state
        container.classList.add('loading');
        container.classList.remove('loaded', 'error');
        
        // Load new image
        const newImg = new Image();
        newImg.onload = () => {
            img.src = path;
            img.classList.remove('hidden');
            if (placeholder) {
                placeholder.classList.add('hidden');
            }
            container.classList.remove('loading');
            container.classList.add('loaded');
        };
        
        newImg.onerror = () => {
            console.error(`Failed to load visualization: ${path}`);
            container.classList.remove('loading');
            container.classList.add('error');
            if (placeholder) {
                placeholder.classList.remove('hidden');
                placeholder.innerHTML = `
                    <svg class="error-icon" width="48" height="48" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                    </svg>
                    <p>Failed to load visualization</p>
                `;
            }
        };
        
        newImg.src = path;
    }

    function startElapsedTimer() {
        elapsedTimer = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const hours = Math.floor(elapsed / 3600000);
            const minutes = Math.floor((elapsed % 3600000) / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            elapsedTime.textContent = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }, 1000);
    }

    function pipelineComplete() {
        appendToLog('Pipeline execution completed successfully!');
        showNotification('Pipeline Complete', 'Analysis has finished successfully', 'success');
        resetPipelineState();
    }

    function handleError(message) {
        appendToLog(`Error: ${message}`);
        showNotification('Pipeline Error', message, 'error');
        resetPipelineState();
    }

    function resetPipelineState() {
        pipelineRunning = false;
        startPipelineBtn.textContent = 'Start Analysis';
        startPipelineBtn.disabled = false;
        clearInterval(elapsedTimer);
    }

    function showNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-header">
                <h4>${title}</h4>
                <button class="notification-close">&times;</button>
            </div>
            <p>${message}</p>
        `;
        
        // Add close button handler
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        });
        
        // Add to document
        document.body.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Auto-remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }

    // Navigation
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    function highlightNavLink() {
        const scrollPosition = window.scrollY + 100;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionBottom = sectionTop + section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    window.addEventListener('scroll', highlightNavLink);
    
    // Initialize WebSocket connection
    setupWebSocket();
});