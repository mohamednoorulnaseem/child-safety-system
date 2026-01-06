// Configuration
const API_BASE_URL = 'http://localhost:5000';
const REFRESH_INTERVAL = 5000; // 5 seconds

// Global variables
let priorityChart = null;
let timelineChart = null;
let currentFilter = 'all';
let refreshTimer = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    initializeCharts();
    loadAllData();
    startAutoRefresh();
    updateClock();
    setInterval(updateClock, 1000);
});

// Update clock
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    const dateString = now.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
    document.getElementById('currentTime').textContent = `${dateString} ${timeString}`;
}

// Load all data
async function loadAllData() {
    await Promise.all([
        loadAlerts(),
        loadCameraStatus(),
        loadStatistics()
    ]);
}

// Refresh all data
function refreshAll() {
    console.log('Refreshing all data...');
    loadAllData();
}

// Start auto-refresh
function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    refreshTimer = setInterval(() => {
        loadAllData();
    }, REFRESH_INTERVAL);
}

// Load alerts
async function loadAlerts() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/alerts`);
        const data = await response.json();
        
        if (data.alerts) {
            displayAlerts(data.alerts);
            updateAlertStats(data.alerts);
        }
        
        updateSystemStatus('online');
    } catch (error) {
        console.error('Error loading alerts:', error);
        updateSystemStatus('offline');
    }
}

// Display alerts in table
function displayAlerts(alerts) {
    const tbody = document.getElementById('alertsBody');
    
    // Filter alerts
    let filteredAlerts = alerts;
    if (currentFilter !== 'all') {
        filteredAlerts = alerts.filter(a => a.priority.toLowerCase() === currentFilter);
    }
    
    // Sort by timestamp (newest first)
    filteredAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    // Display alerts
    if (filteredAlerts.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 40px; color: #999;">No alerts found</td></tr>';
        return;
    }
    
    tbody.innerHTML = filteredAlerts.map(alert => `
        <tr>
            <td>
                <span class="priority-badge priority-${alert.priority.toLowerCase()}">
                    ${alert.priority.toUpperCase()}
                </span>
            </td>
            <td>${escapeHtml(alert.message)}</td>
            <td>${alert.camera_id || 'N/A'}</td>
            <td>${formatTime(alert.timestamp)}</td>
            <td>${alert.status || 'pending'}</td>
            <td>
                ${alert.status !== 'acknowledged' ? `
                    <button class="action-btn btn-acknowledge" onclick="acknowledgeAlert(${alert.id})">
                        ✓ Acknowledge
                    </button>
                ` : ''}
                <button class="action-btn btn-dismiss" onclick="dismissAlert(${alert.id})">
                    ✕ Dismiss
                </button>
            </td>
        </tr>
    `).join('');
}

// Update alert statistics
function updateAlertStats(alerts) {
    const critical = alerts.filter(a => a.priority.toLowerCase() === 'critical').length;
    const high = alerts.filter(a => a.priority.toLowerCase() === 'high').length;
    const medium = alerts.filter(a => a.priority.toLowerCase() === 'medium').length;
    
    document.getElementById('criticalCount').textContent = critical;
    document.getElementById('highCount').textContent = high;
    document.getElementById('mediumCount').textContent = medium;
    
    // Update priority chart
    if (priorityChart) {
        priorityChart.data.datasets[0].data = [critical, high, medium];
        priorityChart.update();
    }
}

// Load camera status
async function loadCameraStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/cameras/status`);
        const data = await response.json();
        
        if (data.cameras) {
            displayCameras(data.cameras);
            
            const onlineCount = data.cameras.filter(c => c.is_online).length;
            document.getElementById('camerasOnline').textContent = `${onlineCount}/${data.cameras.length}`;
        }
    } catch (error) {
        console.error('Error loading camera status:', error);
    }
}

// Display cameras
function displayCameras(cameras) {
    const container = document.getElementById('cameraList');
    
    if (cameras.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #999; padding: 40px;">No cameras detected</p>';
        return;
    }
    
    container.innerHTML = cameras.map(camera => `
        <div class="camera-item">
            <div class="camera-header">
                <span class="camera-name">📹 ${escapeHtml(camera.name || `Camera ${camera.id}`)}</span>
                <span class="camera-status ${camera.is_online ? 'status-online' : 'status-offline'}">
                    ${camera.is_online ? '● Online' : '● Offline'}
                </span>
            </div>
            <div class="camera-info">
                <div>
                    <span>FPS:</span>
                    <span>${camera.fps ? camera.fps.toFixed(1) : '0.0'}</span>
                </div>
                <div>
                    <span>Location:</span>
                    <span>${escapeHtml(camera.location || 'Unknown')}</span>
                </div>
                <div>
                    <span>Trackers:</span>
                    <span>${camera.active_trackers || 0}</span>
                </div>
                <div>
                    <span>Last Detection:</span>
                    <span>${camera.last_detection ? formatTime(camera.last_detection) : 'Never'}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/statistics`);
        const data = await response.json();
        
        // Update timeline chart with hourly data
        if (timelineChart && data.alerts_per_hour) {
            updateTimelineChart(data.alerts_per_hour);
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Initialize charts
function initializeCharts() {
    // Priority Chart
    const priorityCtx = document.getElementById('priorityChart').getContext('2d');
    priorityChart = new Chart(priorityCtx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Medium'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#DC2626', '#F59E0B', '#3B82F6'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                }
            }
        }
    });
    
    // Timeline Chart
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    
    timelineChart = new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{
                label: 'Alerts',
                data: Array(24).fill(0),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            }
        }
    });
}

// Update timeline chart
function updateTimelineChart(data) {
    if (timelineChart && Array.isArray(data)) {
        timelineChart.data.datasets[0].data = data;
        timelineChart.update();
    }
}

// Filter alerts
function filterAlerts(priority) {
    currentFilter = priority;
    
    // Update button states
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Reload alerts
    loadAlerts();
}

// Acknowledge alert
async function acknowledgeAlert(alertId) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/alerts/${alertId}/acknowledge`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log(`Alert ${alertId} acknowledged`);
            loadAlerts();
        } else {
            console.error('Failed to acknowledge alert');
        }
    } catch (error) {
        console.error('Error acknowledging alert:', error);
    }
}

// Dismiss alert
async function dismissAlert(alertId) {
    if (!confirm('Are you sure you want to dismiss this alert?')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/alerts/${alertId}/dismiss`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log(`Alert ${alertId} dismissed`);
            loadAlerts();
        } else {
            console.error('Failed to dismiss alert');
        }
    } catch (error) {
        console.error('Error dismissing alert:', error);
    }
}

// Update system status
function updateSystemStatus(status) {
    const statusEl = document.getElementById('systemStatus');
    statusEl.textContent = status === 'online' ? 'Online' : 'Offline';
    statusEl.className = status === 'online' ? 'status-online' : 'status-offline';
}

// Utility functions
function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000); // seconds
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Handle page visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (refreshTimer) {
            clearInterval(refreshTimer);
        }
    } else {
        startAutoRefresh();
        loadAllData();
    }
});
