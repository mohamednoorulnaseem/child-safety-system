# 👥 Team Setup Guide

Welcome to the **AI-Powered Child Safety System** project! Follow these steps to get started.

## 🚀 Quick Start for Teammates

### Prerequisites
- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Flutter 3.0+** - [Download](https://flutter.dev/docs/get-started/install) _(for mobile app)_
- **Git** - [Download](https://git-scm.com/downloads)
- **VS Code** - [Download](https://code.visualstudio.com/) _(recommended)_

---

## 📥 Step 1: Clone Repository

```bash
git clone https://github.com/mohamednoorulnaseem/child-safety-system.git
cd child-safety-system
```

---

## 🐍 Step 2: Setup Python Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/Mac
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Expected time:** 5-10 minutes (downloads ~2GB of libraries)

---

## 📱 Step 3: Setup Mobile App (Optional)

```bash
cd mobile_app
flutter pub get
flutter analyze
```

**Expected output:** "No issues found!"

---

## ▶️ Step 4: Test the System

### Test Core Detection
```bash
python test_quick.py
```
**Expected:** All tests pass (89% coverage)

### Test with Camera
```bash
python main.py
```
**Expected:** Real-time detection at 25-30 FPS

### Test API Server
```bash
python src/api/app.py
```
**Expected:** Server running on http://localhost:5000

---

## 📂 Project Structure

```
child-safety-system/
├── src/                    # Core Python modules
│   ├── detection/         # YOLOv8 person detection
│   ├── tracking/          # DeepSORT tracking
│   ├── alerts/            # Alert management
│   └── api/               # Flask REST API
├── mobile_app/            # Flutter mobile app
│   ├── lib/               # Dart source code
│   └── assets/            # Images/icons/fonts
├── training/              # Model training scripts
├── web_dashboard/         # Web monitoring dashboard
├── tests/                 # Unit tests
├── models/                # AI model files (.pt)
├── config/                # Configuration files
└── data/                  # Database & logs

```

---

## 🎯 Common Tasks

### Run Detection System
```bash
python main.py
```

### Train Age Model
```bash
cd training
python train_age_model.py
```

### Run Mobile App
```bash
cd mobile_app
flutter run -d chrome    # Run on Chrome
flutter run -d windows   # Run on Windows
```

### View Web Dashboard
```bash
cd web_dashboard
# Open index.html in browser
# Or use: python -m http.server 8080
```

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "CUDA not available"
**Solution:** PyTorch will use CPU (works fine, slightly slower)

### Issue: "Camera not detected"
```bash
python scripts/test_camera.py
```

### Issue: Flutter errors
```bash
cd mobile_app
flutter clean
flutter pub get
flutter analyze
```

---

## 👥 Team Roles

| Member | Role | Focus Area |
|--------|------|------------|
| Mohamed Noorul Naseem | Lead Developer | Core system, integration |
| Mohamed Usman Ali | Backend & API | Flask API, database |
| Kabilash | Mobile Development | Flutter app |
| Manimaran | Hardware Integration | Raspberry Pi, cameras |

---

## 📚 Documentation

- **[README.md](README.md)** - Complete project overview
- **[QUICKSTART.md](QUICKSTART.md)** - Detailed setup guide
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing procedures
- **[FINAL_PROJECT_SUMMARY.md](FINAL_PROJECT_SUMMARY.md)** - Full project stats

---

## 🐛 Reporting Issues

1. Check existing issues: https://github.com/mohamednoorulnaseem/child-safety-system/issues
2. Create new issue with:
   - Clear description
   - Steps to reproduce
   - Error messages
   - System info (OS, Python version)

---

## 🤝 Contributing

### Before committing:
```bash
# Run tests
python test_quick.py

# Check code quality (for Flutter)
cd mobile_app
flutter analyze
```

### Commit message format:
```
feat: Add new feature
fix: Fix bug
docs: Update documentation
chore: Maintenance tasks
```

---

## 📞 Need Help?

**Team Lead:** Mohamed Noorul Naseem  
**GitHub Issues:** [Create Issue](https://github.com/mohamednoorulnaseem/child-safety-system/issues/new)

---

## ✅ System Requirements

### Minimum (Development)
- **CPU:** Intel i3 or equivalent
- **RAM:** 8GB
- **Storage:** 10GB free space
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 11+

### Recommended (Full System)
- **CPU:** Intel i5 or equivalent
- **RAM:** 16GB
- **GPU:** NVIDIA (optional, for faster training)
- **Camera:** USB webcam or Raspberry Pi camera

---

## 🎓 Academic Project Info

- **Institution:** Anand Institute of Higher Technology
- **Department:** AI & Data Science
- **Duration:** 6 months (24 weeks)
- **Status:** ✅ 100% Software Complete
- **Hardware:** Awaiting Raspberry Pi delivery

---

**Last Updated:** January 6, 2026  
**Repository:** https://github.com/mohamednoorulnaseem/child-safety-system
