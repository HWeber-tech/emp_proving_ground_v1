/**
 * EMP v1.0 Dashboard JavaScript
 * Real-time WebSocket connection and UI updates
 */

class EMPDashboard {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.eventCount = 0;
        this.tradeCount = 0;
        this.chaosEnabled = false;
        
        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupPeriodicUpdates();
        this.updateConnectionStatus('Connecting...');
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/events`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('Connected');
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleEvent(data);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus('Disconnected');
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error');
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus('Connection Failed');
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            this.updateConnectionStatus(`Reconnecting... (${this.reconnectAttempts})`);
            
            setTimeout(() => {
                this.setupWebSocket();
            }, this.reconnectInterval);
        } else {
            this.updateConnectionStatus('Connection Failed');
        }
    }

    handleEvent(event) {
        this.eventCount++;
        document.getElementById('event-count').textContent = this.eventCount;
        
        // Update last update time
        const now = new Date().toLocaleTimeString();
        document.getElementById('last-update').textContent = now;
        
        switch (event.type) {
            case 'PortfolioStatus':
                this.updatePortfolio(event.data);
                break;
            case 'MarketData':
                this.updateMarketData(event.data);
                break;
            case 'TradeExecution':
                this.updateTradeLog(event.data);
                break;
            case 'PositionUpdate':
                this.updateActiveTrades(event.data);
                break;
            case 'SystemHealth':
                this.updateSystemHealth(event.data);
                break;
            case 'PatternMemory':
                this.updateMemoryContext(event.data);
                break;
            case 'ChaosEvent':
                this.updateChaosStatus(event.data);
                break;
            default:
                this.addGenericEvent(event);
        }
    }

    updatePortfolio(data) {
        document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
        document.getElementById('equity').textContent = `$${data.equity.toFixed(2)}`;
        
        const pnlElement = document.getElementById('pnl');
        pnlElement.textContent = `$${data.pnl.toFixed(2)}`;
        pnlElement.className = `value ${data.pnl >= 0 ? 'positive' : 'negative'}`;
        
        document.getElementById('win-rate').textContent = `${(data.winRate * 100).toFixed(1)}%`;
        document.getElementById('portfolio-timestamp').textContent = new Date(data.timestamp).toLocaleTimeString();
    }

    updateMarketData(data) {
        document.getElementById('market-symbol').textContent = data.symbol;
        document.getElementById('current-price').textContent = data.price.toFixed(5);
        document.getElementById('spread').textContent = data.spread.toFixed(5);
        document.getElementById('volume').textContent = data.volume.toLocaleString();
        document.getElementById('volatility').textContent = `${(data.volatility * 100).toFixed(2)}%`;
    }

    updateTradeLog(data) {
        this.tradeCount++;
        document.getElementById('trade-count').textContent = this.tradeCount;
        
        const tbody = document.getElementById('trade-log-tbody');
        const noTradesRow = tbody.querySelector('.no-trades');
        if (noTradesRow) {
            noTradesRow.remove();
        }
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${new Date(data.timestamp).toLocaleTimeString()}</td>
            <td>${data.action}</td>
            <td>${data.symbol}</td>
            <td>${data.volume}</td>
            <td>${data.price.toFixed(5)}</td>
            <td class="${data.pnl >= 0 ? 'positive' : 'negative'}">${data.pnl.toFixed(2)}</td>
        `;
        
        tbody.insertBefore(row, tbody.firstChild);
        
        // Keep only last 50 trades
        while (tbody.children.length > 50) {
            tbody.removeChild(tbody.lastChild);
        }
    }

    updateActiveTrades(data) {
        const tbody = document.getElementById('trades-tbody');
        const noTradesRow = tbody.querySelector('.no-trades');
        if (noTradesRow) {
            noTradesRow.remove();
        }
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Update active trades count
        document.getElementById('active-trades').textContent = data.trades.length;
        
        if (data.trades.length === 0) {
            const row = document.createElement('tr');
            row.className = 'no-trades';
            row.innerHTML = '<td colspan="7">No active trades</td>';
            tbody.appendChild(row);
            return;
        }
        
        data.trades.forEach(trade => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${trade.id}</td>
                <td>${trade.symbol}</td>
                <td>${trade.type}</td>
                <td>${trade.volume}</td>
                <td>${trade.price.toFixed(5)}</td>
                <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">${trade.pnl.toFixed(2)}</td>
                <td>${new Date(trade.openTime).toLocaleTimeString()}</td>
            `;
            tbody.appendChild(row);
        });
    }

    updateSystemHealth(data) {
        document.getElementById('cpu-usage').textContent = `${data.cpuUsage.toFixed(1)}%`;
        document.getElementById('memory-usage').textContent = `${data.memoryUsage.toFixed(0)}MB`;
        
        const redisStatus = document.getElementById('redis-status');
        redisStatus.textContent = data.redisStatus;
        redisStatus.className = data.redisStatus === 'OK' ? 'status-ok' : 'status-error';
        
        const dbStatus = document.getElementById('db-status');
        dbStatus.textContent = data.dbStatus;
        dbStatus.className = data.dbStatus === 'OK' ? 'status-ok' : 'status-error';
        
        document.getElementById('health-timestamp').textContent = new Date(data.timestamp).toLocaleTimeString();
    }

    updateMemoryContext(data) {
        document.getElementById('similar-patterns').textContent = data.similarPatterns;
        document.getElementById('avg-pnl').textContent = `$${data.averagePnl.toFixed(2)}`;
        document.getElementById('memory-win-rate').textContent = `${(data.winRate * 100).toFixed(1)}%`;
        document.getElementById('memory-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    }

    updateChaosStatus(data) {
        this.chaosEnabled = data.active;
        document.getElementById('chaos-active').textContent = data.active ? 'Active' : 'Inactive';
        document.getElementById('chaos-active').className = data.active ? 'status-ok' : 'status-inactive';
        document.getElementById('chaos-events').textContent = data.eventCount;
    }

    addGenericEvent(event) {
        const container = document.getElementById('event-stream');
        const loading = container.querySelector('.loading');
        if (loading) {
            loading.remove();
        }
        
        const eventDiv = document.createElement('div');
        eventDiv.className = `event-item ${event.level || 'info'}`;
        eventDiv.innerHTML = `
            <div class="event-time">${new Date().toLocaleTimeString()}</div>
            <div class="event-message">${event.message || JSON.stringify(event)}</div>
        `;
        
        container.insertBefore(eventDiv, container.firstChild);
        
        // Keep only last 100 events
        while (container.children.length > 100) {
            container.removeChild(container.lastChild);
        }
    }

    updateConnectionStatus(status) {
        const indicator = document.getElementById('connection-status');
        indicator.textContent = status;
        indicator.className = `status-indicator ${status === 'Connected' ? 'connected' : 'disconnected'}`;
    }

    setupEventListeners() {
        // Chaos testing controls
        document.getElementById('chaos-enable').addEventListener('click', () => {
            this.toggleChaos(true);
        });
        
        document.getElementById('chaos-disable').addEventListener('click', () => {
            this.toggleChaos(false);
        });
        
        // Auto-refresh on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.ws && this.ws.readyState === WebSocket.CLOSED) {
                this.setupWebSocket();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshData();
                        break;
                    case 'k':
                        e.preventDefault();
                        this.clearEventStream();
                        break;
                }
            }
        });
    }

    setupPeriodicUpdates() {
        // Request health updates every 30 seconds
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'RequestHealthUpdate' }));
            }
        }, 30000);
        
        // Request memory context updates every 60 seconds
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'RequestMemoryContext' }));
            }
        }, 60000);
    }

    toggleChaos(enable) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'ToggleChaos',
                data: { active: enable }
            }));
        }
    }

    refreshData() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'RefreshAll' }));
        }
    }

    clearEventStream() {
        const container = document.getElementById('event-stream');
        container.innerHTML = '<div class="loading">Event stream cleared</div>';
        this.eventCount = 0;
        document.getElementById('event-count').textContent = this.eventCount;
    }

    // Utility methods
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }

    formatNumber(number) {
        return new Intl.NumberFormat('en-US').format(number);
    }

    formatPercentage(value) {
        return `${(value * 100).toFixed(2)}%`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new EMPDashboard();
});

// Handle page visibility for mobile
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        window.dashboard.setupPeriodicUpdates();
    }
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EMPDashboard;
}
