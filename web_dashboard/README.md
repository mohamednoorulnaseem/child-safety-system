# 🌐 Web Dashboard - Child Safety System

Real-time monitoring dashboard for the Child Safety System.

## 📊 Features

- **Real-time Alerts**: View all system alerts with priority filtering
- **Camera Status**: Monitor all cameras with FPS and health metrics
- **Interactive Charts**: Visualize alert trends and priority distribution
- **Auto-refresh**: Updates every 5 seconds automatically
- **Responsive Design**: Works on desktop, tablet, and mobile

## 🚀 Quick Start

### Method 1: Open Directly
1. Ensure API server is running:
   ```bash
   python src/api/app.py
   ```

2. Open `index.html` in your browser:
   - Double-click the file, OR
   - Right-click → Open with → Your browser

### Method 2: Using HTTP Server
```bash
# Navigate to dashboard folder
cd web_dashboard

# Start Python HTTP server
python -m http.server 8080

# Open browser to:
http://localhost:8080
```

## 📁 Files

- **index.html** - Dashboard structure and layout
- **styles.css** - Styling and responsive design
- **dashboard.js** - API integration and real-time updates

## ⚙️ Configuration

Edit `dashboard.js` to change settings:

```javascript
const API_BASE_URL = 'http://localhost:5000';  // Change for deployment
const REFRESH_INTERVAL = 5000;  // Update frequency (milliseconds)
```

## 🎨 Dashboard Sections

### 1. Stats Cards
- Critical alerts count (red)
- High priority alerts (orange)
- Medium priority alerts (blue)
- Cameras online status (green)

### 2. Charts
- **Priority Distribution**: Doughnut chart showing alert breakdown
- **Timeline**: Line graph of alerts over last 24 hours

### 3. Camera Status
- Real-time FPS monitoring
- Online/offline status
- Active trackers count
- Last detection time

### 4. Recent Alerts
- Sortable table with all alerts
- Filter by priority (All, Critical, High, Medium)
- Actions: Acknowledge and Dismiss
- Color-coded priority badges

## 🔌 API Endpoints Used

```
GET  /api/alerts          - Fetch all alerts
GET  /api/cameras/status  - Get camera status
GET  /api/statistics      - Get statistics
POST /api/alerts/:id/acknowledge - Acknowledge alert
POST /api/alerts/:id/dismiss     - Dismiss alert
```

## 🎯 Usage Tips

1. **Filtering**: Use priority buttons to filter alerts
2. **Refresh**: Click refresh button or wait for auto-update
3. **Actions**: Acknowledge important alerts, dismiss false alarms
4. **Mobile**: Dashboard is fully responsive on phones/tablets

## 🐛 Troubleshooting

### Dashboard shows "Offline"
- Check if API server is running on port 5000
- Verify `API_BASE_URL` in dashboard.js
- Check browser console for errors (F12)

### No data displayed
- Ensure API endpoints are accessible
- Check CORS settings in Flask app
- Verify database has data

### Auto-refresh not working
- Check browser console for JavaScript errors
- Ensure page is visible (refresh pauses on hidden tabs)

## 🔐 Security Notes

For production deployment:
- Change `API_BASE_URL` to your server IP/domain
- Add authentication to API endpoints
- Use HTTPS for secure communication
- Implement rate limiting

## 📱 Mobile Support

Dashboard automatically adapts to smaller screens:
- Stats cards stack vertically
- Charts resize responsively
- Tables scroll horizontally
- Touch-friendly buttons

## 🎨 Customization

### Change Colors
Edit `styles.css`:
```css
.stat-card.critical { border-left: 5px solid #YOUR_COLOR; }
```

### Modify Refresh Rate
Edit `dashboard.js`:
```javascript
const REFRESH_INTERVAL = 10000; // 10 seconds
```

### Add New Charts
Use Chart.js documentation:
https://www.chartjs.org/docs/latest/

## 📸 Screenshots

*Take screenshots after running and add them here*

## 🔗 Related Files

- API Server: `src/api/app.py`
- Alert Manager: `src/alerts/alert_manager.py`
- Database: `data/alerts.db`

---

**Status**: ✅ Production Ready  
**Last Updated**: January 6, 2026
