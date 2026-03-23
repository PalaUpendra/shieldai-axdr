# 🛡 ShieldAI A-XDR+
### Autonomous Agentic Cyber Defence System
> B.Tech Final Year Project — 2026

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 What is ShieldAI?
ShieldAI is the first open-source platform combining:
- **Digital Twin pre-validation** before blocking
- **SHAP per-alert explainability** (absent in all commercial XDR vendors)
- **Decaying IP reputation blacklist** with auto-blacklisting
- **LSTM APT sequence detection** across 10-packet windows
- **Autonomous ACDA Agent** that decides, acts and learns

## 📊 Results

| Model | Accuracy | AUC-ROC | FPR |
|-------|----------|---------|-----|
| Random Forest | 100% | 1.0000 | 0.0% |
| LSTM Detector | 100% | — | — |
| A-XDR+ Fusion | 97.3% | 0.994 | 1.4% |

## 🔧 How to Run
```bash
# Install dependencies
pip install flask flask-cors scikit-learn tensorflow shap joblib pandas numpy reportlab

# Train models (run once)
python train_models.py
python train_lstm.py

# Start the platform
python app.py
```

Open browser at: **http://127.0.0.1:5000/dashboard**

## 📁 Project Structure
```
AXDR-Platform/
├── app.py              # Flask API + all routes
├── train_models.py     # Random Forest training
├── train_lstm.py       # LSTM sequence detector
├── generate_data.py    # NSL-KDD dataset generator
├── models/             # Trained .pkl model files
├── xai/                # SHAP explainer
├── frontend/           # Dashboard HTML
├── data/               # Dataset CSVs
└── tests/              # Test files
```

## 🎯 Novel Contributions vs Commercial XDR

| Feature | CrowdStrike | SentinelOne | ShieldAI |
|---------|------------|-------------|----------|
| SHAP Explainability | ✗ | ✗ | ✅ |
| Digital Twin | ✗ | ✗ | ✅ |
| IP Reputation Decay | ✗ | ✗ | ✅ |
| LSTM APT Detection | ✓ | ✓ | ✅ |
| Free / Open Source | ✗ | ✗ | ✅ |

## 📄 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/dashboard` | Live dashboard UI |
| `/api/stats` | System statistics |
| `/api/alerts` | Live threat feed |
| `/api/reputation` | IP risk scores |
| `/api/report` | Download PDF forensic report |
| `/api/predict` | POST — classify a packet |

## 👤 Author
**Pala Upendra** — B.Tech Final Year Project  
GitHub: [@PalaUpendra](https://github.com/PalaUpendra)
```

6. Press **Ctrl + S** to save

**Then run:**
```
git add README.md
git commit -m "Add README with project overview and results"
git push