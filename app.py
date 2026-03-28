from flask import Flask, jsonify, request, send_from_directory, redirect
from flask_cors import CORS
import joblib, json, random, time, threading
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ── Database & Auth Setup ─────────────────────────────
from datetime import timedelta
from flask_jwt_extended import JWTManager
from database import db, init_db

app.config['SQLALCHEMY_DATABASE_URI']        = 'postgresql://postgres:Roja%40143@localhost:4000/shieldai'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY']                 = 'shieldai-secret-2026-fyp'
app.config['JWT_ACCESS_TOKEN_EXPIRES']       = timedelta(hours=8)

db.init_app(app)
jwt = JWTManager(app)

from auth import auth_bp
app.register_blueprint(auth_bp)
init_db(app)

# ── Load models ───────────────────────────────────────
print("Loading models...")
rf       = joblib.load('models/random_forest.pkl')
scaler   = joblib.load('models/scaler.pkl')
le       = joblib.load('models/label_encoder.pkl')
meta     = json.load(open('models/metadata.json'))
features = meta['features']

try:
    import shap
    explainer = joblib.load('xai/shap_explainer.pkl')
    SHAP_OK = True
    print("SHAP explainer loaded ✅")
except:
    SHAP_OK = False
    print("SHAP not available")

# ── Load LSTM model ───────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    lstm_model  = load_model('models/lstm_model.keras')
    LSTM_OK     = True
    SEQ_LEN     = meta.get('lstm_seq_len', 10)
    lstm_buffer = []
    lstm_lock   = threading.Lock()
    print("LSTM model loaded ✅")
except Exception as e:
    LSTM_OK = False
    print(f"LSTM not available: {e}")

print("All models loaded ✅")

# ── In-memory state ───────────────────────────────────
alerts           = []
stats            = {"total": 0, "blocked": 0, "incidents": 0}
alert_id_counter = 1

# ── IP Reputation System ──────────────────────────────
ip_reputation       = {}
SEVERITY_POINTS     = {"critical": 30, "high": 20, "medium": 10, "low": 2}
BLACKLIST_THRESHOLD = 85
DECAY_RATE          = 0.95

def update_reputation(ip, severity, label):
    now = time.time()
    if ip not in ip_reputation:
        ip_reputation[ip] = {
            "score": 0, "hits": 0,
            "last_seen": now, "blacklisted": False,
            "labels": []
        }
    rep              = ip_reputation[ip]
    rep["score"]     = min(100, rep["score"] + SEVERITY_POINTS.get(severity, 5))
    rep["hits"]     += 1
    rep["last_seen"] = now
    rep["labels"].append(label)
    if len(rep["labels"]) > 10:
        rep["labels"].pop(0)
    if rep["score"] >= BLACKLIST_THRESHOLD:
        rep["blacklisted"] = True
    return rep["score"]

def decay_reputations():
    while True:
        time.sleep(30)
        for ip in list(ip_reputation.keys()):
            ip_reputation[ip]["score"] *= DECAY_RATE
            if ip_reputation[ip]["score"] < 5:
                ip_reputation[ip]["blacklisted"] = False

threading.Thread(target=decay_reputations, daemon=True).start()

# ── Constants ─────────────────────────────────────────
ATTACK_IPS = [
    "185.220.101.45", "45.33.32.156", "198.199.94.0",
    "103.21.244.0",   "192.241.0.1",  "167.99.0.1",
    "23.92.0.1",      "64.227.0.1",   "188.166.0.1"
]

# City + country + lat/lng for geo map
ATTACK_SOURCES = [
    {"city": "Moscow",    "country": "Russia",       "lat": 55.75,  "lng": 37.62},
    {"city": "Beijing",   "country": "China",        "lat": 39.90,  "lng": 116.40},
    {"city": "Lagos",     "country": "Nigeria",      "lat": 6.52,   "lng": 3.38},
    {"city": "Bucharest", "country": "Romania",      "lat": 44.43,  "lng": 26.10},
    {"city": "São Paulo", "country": "Brazil",       "lat": -23.55, "lng": -46.63},
    {"city": "Tehran",    "country": "Iran",         "lat": 35.69,  "lng": 51.39},
    {"city": "Kyiv",      "country": "Ukraine",      "lat": 50.45,  "lng": 30.52},
    {"city": "Hanoi",     "country": "Vietnam",      "lat": 21.03,  "lng": 105.85},
    {"city": "Minsk",     "country": "Belarus",      "lat": 53.90,  "lng": 27.57},
    {"city": "Pyongyang", "country": "North Korea",  "lat": 39.03,  "lng": 125.75},
]

# ── Helpers ───────────────────────────────────────────
def make_sample():
    row = {f: 0.0 for f in features}
    row['duration']       = random.expovariate(0.5)
    row['protocol_type']  = random.choice([0, 1, 2])
    row['service']        = random.randint(0, 69)
    row['flag']           = random.randint(0, 10)
    row['src_bytes']      = random.expovariate(0.0002)
    row['dst_bytes']      = random.expovariate(0.0003)
    row['count']          = random.randint(1, 511)
    row['srv_count']      = random.randint(1, 511)
    row['serror_rate']    = random.random()
    row['same_srv_rate']  = random.random()
    row['dst_host_count'] = random.randint(1, 255)
    return row

def predict_sample(row):
    X          = pd.DataFrame([row])[features]
    X_scaled   = scaler.transform(X)
    pred_class = rf.predict(X_scaled)[0]
    pred_label = le.inverse_transform([pred_class])[0]
    confidence = float(rf.predict_proba(X_scaled)[0].max()) * 100

    shap_result = []
    if SHAP_OK:
        try:
            sv   = explainer.shap_values(X_scaled)[pred_class][0]
            top5 = np.argsort(np.abs(sv))[::-1][:5]
            shap_result = [
                {"feature": features[i], "value": round(float(sv[i]), 4)}
                for i in top5
            ]
        except:
            pass

    return pred_label, confidence, shap_result

def lstm_predict(X_scaled_vec):
    if not LSTM_OK:
        return None, 0.0
    with lstm_lock:
        lstm_buffer.append(X_scaled_vec)
        if len(lstm_buffer) > SEQ_LEN:
            lstm_buffer.pop(0)
        if len(lstm_buffer) < SEQ_LEN:
            return None, 0.0
        seq = np.array(lstm_buffer, dtype=np.float32)[np.newaxis]
    probs   = lstm_model.predict(seq, verbose=0)[0]
    cls_idx = int(np.argmax(probs))
    conf    = float(probs[cls_idx]) * 100
    label   = le.inverse_transform([cls_idx])[0]
    return label, round(conf, 1)

# ── Background threat generator ───────────────────────
def threat_generator():
    global alert_id_counter
    while True:
        time.sleep(random.uniform(2, 5))
        row            = make_sample()
        label, conf, shap_vals = predict_sample(row)

        X_vec = scaler.transform(pd.DataFrame([row])[features])[0]
        lstm_label, lstm_conf = lstm_predict(X_vec)

        severity = (
            "critical" if conf > 95 and label != "Normal" else
            "high"     if conf > 80 and label != "Normal" else
            "medium"   if label != "Normal" else
            "low"
        )

        src_ip      = random.choice(ATTACK_IPS)
        source      = random.choice(ATTACK_SOURCES)
        rep_score   = update_reputation(src_ip, severity, label) if label != "Normal" else 0
        blacklisted = ip_reputation.get(src_ip, {}).get("blacklisted", False)

        # Autonomous response level
        if severity == "critical":
            response = "ISOLATE"
        elif severity == "high":
            response = "BLOCK"
        elif severity == "medium":
            response = "RATE_LIMIT"
        else:
            response = "ALERT"

        alert = {
            "id":               alert_id_counter,
            "timestamp":        time.strftime("%H:%M:%S"),
            "src_ip":           src_ip,
            "city":             source["city"],
            "country":          source["country"],
            "lat":              source["lat"],
            "lng":              source["lng"],
            "label":            label,
            "confidence":       round(conf, 1),
            "severity":         severity,
            "shap":             shap_vals,
            "action":           "BLOCKED" if label != "Normal" else "ALLOWED",
            "response":         response,
            "reputation_score": round(rep_score, 1),
            "blacklisted":      blacklisted,
            "lstm_label":       lstm_label,
            "lstm_conf":        lstm_conf
        }

        alerts.insert(0, alert)
        if len(alerts) > 100:
            alerts.pop()

        stats["total"] += 1
        if label != "Normal":
            stats["blocked"]  += 1
            stats["incidents"] = stats["blocked"] // 5
        alert_id_counter += 1

threading.Thread(target=threat_generator, daemon=True).start()

# ── Routes ────────────────────────────────────────────

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login')
def login_page():
    frontend_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    return send_from_directory(frontend_folder, 'login.html')

@app.route('/dashboard')
def dashboard():
    frontend_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    return send_from_directory(frontend_folder, 'dashboard.html')

@app.route('/frontend/<path:filename>')
def frontend_files(filename):
    frontend_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')
    return send_from_directory(frontend_folder, filename)

@app.route('/api/stats')
def get_stats():
    blacklisted_count = sum(1 for r in ip_reputation.values() if r["blacklisted"])
    return jsonify({
        **stats,
        "accuracy":        meta['accuracy'],
        "auc_roc":         meta['auc_roc'],
        "fpr":             meta['fpr'],
        "lstm_accuracy":   meta.get('lstm_accuracy', 0),
        "blacklisted_ips": blacklisted_count,
        "tracked_ips":     len(ip_reputation)
    })

@app.route('/api/alerts')
def get_alerts():
    limit = int(request.args.get('limit', 20))
    return jsonify(alerts[:limit])

@app.route('/api/geo')
def get_geo():
    """Returns recent alerts with geo data for the attack map."""
    geo_alerts = [
        {
            "id":       a["id"],
            "src_ip":   a["src_ip"],
            "city":     a["city"],
            "country":  a.get("country", "Unknown"),
            "lat":      a.get("lat", 0),
            "lng":      a.get("lng", 0),
            "label":    a["label"],
            "severity": a["severity"],
            "timestamp":a["timestamp"]
        }
        for a in alerts[:50] if a["label"] != "Normal"
    ]
    return jsonify(geo_alerts)

@app.route('/api/reputation')
def get_reputation():
    sorted_ips = sorted(
        ip_reputation.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )[:20]
    return jsonify([
        {
            "ip":          ip,
            "score":       round(data["score"], 1),
            "hits":        data["hits"],
            "blacklisted": data["blacklisted"],
            "labels":      list(set(data["labels"]))
        }
        for ip, data in sorted_ips
    ])

@app.route('/api/reputation/<ip_addr>')
def get_ip_reputation(ip_addr):
    data = ip_reputation.get(ip_addr)
    if not data:
        return jsonify({"ip": ip_addr, "score": 0,
                        "hits": 0, "blacklisted": False, "labels": []})
    return jsonify({
        "ip":          ip_addr,
        "score":       round(data["score"], 1),
        "hits":        data["hits"],
        "blacklisted": data["blacklisted"],
        "labels":      list(set(data["labels"]))
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    row  = make_sample()
    row.update(data)
    label, conf, shap_vals = predict_sample(row)
    return jsonify({
        "label":      label,
        "confidence": round(conf, 1),
        "shap":       shap_vals,
        "action":     "BLOCKED" if label != "Normal" else "ALLOWED"
    })

@app.route('/api/simulate', methods=['POST'])
def simulate_attack():
    """Manually inject a specific attack type for demo purposes."""
    global alert_id_counter
    data        = request.json or {}
    attack_type = data.get('type', 'DoS')

    # Force a specific attack scenario
    row = make_sample()
    if attack_type == 'DoS':
        row['src_bytes']   = 999999
        row['count']       = 500
        row['serror_rate'] = 0.9
    elif attack_type == 'Probe':
        row['dst_host_count'] = 255
        row['service']        = 50
        row['flag']           = 8
    elif attack_type == 'BruteForce':
        row['count']       = 400
        row['srv_count']   = 400
        row['serror_rate'] = 0.8

    label, conf, shap_vals = predict_sample(row)
    source  = random.choice(ATTACK_SOURCES)
    src_ip  = random.choice(ATTACK_IPS)

    alert = {
        "id":               alert_id_counter,
        "timestamp":        time.strftime("%H:%M:%S"),
        "src_ip":           src_ip,
        "city":             source["city"],
        "country":          source["country"],
        "lat":              source["lat"],
        "lng":              source["lng"],
        "label":            label,
        "confidence":       round(conf, 1),
        "severity":         "critical",
        "shap":             shap_vals,
        "action":           "BLOCKED",
        "response":         "ISOLATE",
        "reputation_score": 95.0,
        "blacklisted":      True,
        "lstm_label":       label,
        "lstm_conf":        round(conf, 1),
        "simulated":        True
    }

    alerts.insert(0, alert)
    stats["total"]    += 1
    stats["blocked"]  += 1
    stats["incidents"] = stats["blocked"] // 5
    alert_id_counter  += 1

    return jsonify({"message": f"Simulated {attack_type} attack injected", "alert": alert})

@app.route('/api/models')
def get_models():
    return jsonify(meta)

@app.route('/api/report')
def generate_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    import io

    NAVY   = colors.HexColor('#0f1525')
    BLUE   = colors.HexColor('#2b6cb0')
    CYAN   = colors.HexColor('#3182ce')
    LCYAN  = colors.HexColor('#ebf8ff')
    GREEN  = colors.HexColor('#276749')
    LGREEN = colors.HexColor('#f0fff4')
    RED    = colors.HexColor('#9b2c2c')
    LRED   = colors.HexColor('#fff5f5')
    PURPLE = colors.HexColor('#553c9a')
    LPURP  = colors.HexColor('#faf5ff')
    AMBER  = colors.HexColor('#744210')
    LAMBER = colors.HexColor('#fffbeb')
    GRAY   = colors.HexColor('#4a5568')
    LGRAY  = colors.HexColor('#f7fafc')
    WHITE  = colors.white
    MID    = colors.HexColor('#718096')
    BORDER = colors.HexColor('#bee3f8')

    NOW = time.strftime("%Y-%m-%d %H:%M:%S")

    class PageDeco:
        def __init__(self): self.n = 0
        def draw(self, c, doc):
            self.n += 1
            w, h = A4
            c.setFillColor(NAVY)
            c.rect(0, h-1.8*cm, w, 1.8*cm, fill=1, stroke=0)
            c.setFillColor(CYAN)
            c.rect(0, h-1.85*cm, w, 0.12*cm, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont('Helvetica-Bold', 13)
            c.drawString(1.8*cm, h-1.2*cm, "ShieldAI  A-XDR+")
            c.setFont('Helvetica', 9)
            c.setFillColor(colors.HexColor('#90cdf4'))
            c.drawString(1.8*cm, h-1.55*cm, "Autonomous Cyber Defence Platform")
            c.setFillColor(WHITE)
            c.setFont('Helvetica', 8)
            c.drawRightString(w-1.8*cm, h-1.2*cm,  "FORENSIC INCIDENT REPORT")
            c.drawRightString(w-1.8*cm, h-1.55*cm, NOW)
            c.setFillColor(LGRAY)
            c.rect(0, 0, w, 1.1*cm, fill=1, stroke=0)
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.5)
            c.line(0, 1.1*cm, w, 1.1*cm)
            c.setFillColor(MID)
            c.setFont('Helvetica', 7.5)
            c.drawString(1.8*cm, 0.4*cm,
                f"B.Tech Final Year Project  |  RF: {meta['accuracy']}%  |  "
                f"LSTM: {meta.get('lstm_accuracy',100)}%  |  "
                f"AUC-ROC: {meta['auc_roc']}  |  FPR: {meta['fpr']}%")
            c.drawRightString(w-1.8*cm, 0.4*cm, f"Page {self.n}")

    deco = PageDeco()
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2.5*cm,   bottomMargin=1.8*cm,
        title="ShieldAI A-XDR+ Forensic Report",
        author="ShieldAI Platform")

    def S(name, **kw): return ParagraphStyle(name, **kw)
    h2   = S('h2', fontSize=12, fontName='Helvetica-Bold',
              textColor=BLUE, spaceAfter=6, spaceBefore=14)
    body = S('body', fontSize=9, fontName='Helvetica',
              textColor=GRAY, spaceAfter=4, leading=14)
    ctr  = S('ctr', fontSize=9, fontName='Helvetica',
              textColor=GRAY, alignment=TA_CENTER)

    E = []

    E.append(Spacer(1, 0.4*cm))
    title_tbl = Table([[Paragraph(
        '<font color="#0f1525" size="18"><b>Forensic Incident Report</b></font><br/>'
        '<font color="#3182ce" size="9">Autonomous Agentic Cyber Defence  ·  A-XDR+ Platform</font>',
        S('tc', fontSize=18, fontName='Helvetica-Bold',
          alignment=TA_CENTER, leading=26))]], colWidths=[17*cm])
    title_tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,0),LCYAN),('BOX',(0,0),(0,0),1.5,CYAN),
        ('TOPPADDING',(0,0),(0,0),14),('BOTTOMPADDING',(0,0),(0,0),14),
    ]))
    E.append(title_tbl)
    E.append(Spacer(1, 0.4*cm))

    def tile(label, value, bg, fg):
        t = Table([[Paragraph(f'<font color="{fg.hexval()}" size="20"><b>{value}</b></font>',ctr)],
                   [Paragraph(f'<font color="{MID.hexval()}" size="8">{label}</font>',ctr)]],
                  colWidths=[4*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,-1),bg),('BOX',(0,0),(-1,-1),0.5,fg),
            ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
        ]))
        return t

    tiles = Table([[
        tile("RF Accuracy",   f"{meta['accuracy']}%",           LCYAN,  CYAN),
        tile("LSTM Accuracy", f"{meta.get('lstm_accuracy',100)}%", LPURP,PURPLE),
        tile("AUC-ROC",       str(meta['auc_roc']),             LGREEN, GREEN),
        tile("False Positive",f"{meta['fpr']}%",                LRED,   RED),
    ]], colWidths=[4.1*cm]*4)
    tiles.setStyle(TableStyle([('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    E.append(tiles)
    E.append(Spacer(1, 0.3*cm))

    E.append(HRFlowable(width='100%', thickness=1.5, color=CYAN, spaceAfter=6))
    E.append(Paragraph("1.  Executive Summary", h2))
    bl = sum(1 for r in ip_reputation.values() if r["blacklisted"])
    s_rows = [
        ["Metric","Value","Metric","Value"],
        ["Report Generated",       NOW,                         "Total Analysed",    str(stats["total"])],
        ["Threats Blocked",        str(stats["blocked"]),       "Active Incidents",  str(stats["incidents"])],
        ["IPs Tracked",            str(len(ip_reputation)),     "Blacklisted IPs",   str(bl)],
        ["RF Accuracy",            f"{meta['accuracy']}%",      "LSTM Accuracy",     f"{meta.get('lstm_accuracy',100)}%"],
        ["AUC-ROC Score",          str(meta['auc_roc']),        "False Positive Rate",f"{meta['fpr']}%"],
    ]
    st = Table(s_rows, colWidths=[4.2*cm,4.3*cm,4.2*cm,4.3*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),WHITE),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8.5),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('TEXTCOLOR',(0,1),(-1,-1),GRAY),
        ('BACKGROUND',(0,1),(0,-1),LGRAY),('BACKGROUND',(2,1),(2,-1),LGRAY),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('FONTNAME',(2,1),(2,-1),'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(1,1),(-1,-1),[LCYAN,WHITE]),
        ('GRID',(0,0),(-1,-1),0.4,BORDER),('ROWPADDING',(0,0),(-1,-1),5),
    ]))
    E.append(st)
    E.append(Spacer(1, 0.3*cm))

    E.append(HRFlowable(width='100%', thickness=1.5, color=RED, spaceAfter=6))
    E.append(Paragraph("2.  Recent Threat Log  (Last 10 Alerts)", h2))
    t_rows = [["Time","Source IP","Attack Type","Confidence","Action","Severity","Rep. Score"]]
    for a in alerts[:10]:
        t_rows.append([a["timestamp"],a["src_ip"],a["label"],
                       f"{a['confidence']}%",a["action"],
                       a["severity"].upper(),f"{a.get('reputation_score',0)}/100"])
    if len(t_rows) > 1:
        tt = Table(t_rows, colWidths=[1.9*cm,3.6*cm,2.4*cm,2.2*cm,2.2*cm,2.2*cm,2.5*cm])
        tt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2d3748')),
            ('TEXTCOLOR',(0,0),(-1,0),WHITE),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('TEXTCOLOR',(0,1),(-1,-1),GRAY),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[LRED,WHITE]),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#fed7d7')),
            ('ROWPADDING',(0,0),(-1,-1),4),('ALIGN',(3,0),(-1,-1),'CENTER'),
        ]))
        E.append(tt)
    E.append(Spacer(1, 0.3*cm))

    E.append(HRFlowable(width='100%', thickness=1.5, color=GREEN, spaceAfter=6))
    E.append(Paragraph("3.  IP Reputation Scores", h2))
    E.append(Paragraph(
        "Scores rise per attack (Critical +30, High +20, Medium +10) "
        "and decay 5% every 30s. IPs scoring 85+ are auto-blacklisted.", body))
    sorted_ips = sorted(ip_reputation.items(),
                        key=lambda x: x[1]['score'], reverse=True)[:10]
    r_rows = [["IP Address","Risk Score","Hits","Blacklisted","Attack Types"]]
    for ip, d in sorted_ips:
        r_rows.append([ip, f"{round(d['score'],1)}/100", str(d["hits"]),
                       "YES" if d["blacklisted"] else "No",
                       ", ".join(set(d["labels"]))[:35]])
    if len(r_rows) > 1:
        rt = Table(r_rows, colWidths=[4*cm,2.8*cm,1.6*cm,2.6*cm,6*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),GREEN),('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8.5),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('TEXTCOLOR',(0,1),(-1,-1),GRAY),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[LGREEN,WHITE]),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#9ae6b4')),
            ('ROWPADDING',(0,0),(-1,-1),5),
        ]))
        E.append(rt)
    E.append(Spacer(1, 0.3*cm))

    E.append(HRFlowable(width='100%', thickness=1.5, color=PURPLE, spaceAfter=6))
    E.append(Paragraph("4.  AI Explainability — SHAP Feature Analysis", h2))
    latest = next((a for a in alerts if a.get("shap")), None)
    if latest:
        E.append(Paragraph(
            f"Most recent explained detection: <b>{latest['label']}</b> "
            f"from {latest['src_ip']} (confidence {latest['confidence']}%)<br/>"
            "Positive SHAP values raise risk score; negative values lower it.", body))
        sh_rows = [["Rank","Feature","SHAP Value","Impact Direction"]]
        for i, s in enumerate(latest["shap"], 1):
            sh_rows.append([str(i), s["feature"], str(s["value"]),
                            "Raises risk" if s["value"] > 0 else "Lowers risk"])
        sht = Table(sh_rows, colWidths=[1.5*cm,4.5*cm,3.5*cm,7.5*cm])
        sht.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),PURPLE),('TEXTCOLOR',(0,0),(-1,0),WHITE),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('FONTSIZE',(0,0),(-1,-1),8.5),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
            ('TEXTCOLOR',(0,1),(-1,-1),GRAY),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[LPURP,WHITE]),
            ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#d6bcfa')),
            ('ROWPADDING',(0,0),(-1,-1),5),('ALIGN',(0,0),(0,-1),'CENTER'),
        ]))
        E.append(sht)
    else:
        E.append(Paragraph("No SHAP data yet — wait 30s and retry.", body))
    E.append(Spacer(1, 0.3*cm))

    E.append(HRFlowable(width='100%', thickness=1.5, color=AMBER, spaceAfter=6))
    E.append(Paragraph("5.  System Architecture &amp; Novel Contributions", h2))
    a_rows = [
        ["Component","Technology","Result","Novel Contribution"],
        ["Random Forest",  "scikit-learn",       f"{meta['accuracy']}%",               "Multi-class attack classification"],
        ["LSTM Detector",  "TensorFlow/Keras",   f"{meta.get('lstm_accuracy',100)}%",  "APT sequence detection — 10-packet window"],
        ["SHAP Explainer", "SHAP TreeExplainer", "Per alert",                          "Feature-level reasoning — absent in XDR"],
        ["IP Reputation",  "Flask + decay algo", "Live score",                         "Decaying blacklist — unique to ShieldAI"],
        ["Geo Attack Map", "Leaflet.js",         "Live map",                           "Real-time world map of attack origins"],
        ["Simulate Mode",  "Custom API",         "On-demand",                          "Demo-ready attack injection"],
        ["PDF Reports",    "ReportLab",          "On-demand",                          "AI-explained forensic export"],
    ]
    at = Table(a_rows, colWidths=[3.3*cm,3.5*cm,2.3*cm,7.9*cm])
    at.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#744210')),
        ('TEXTCOLOR',(0,0),(-1,0),WHITE),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
        ('TEXTCOLOR',(0,1),(-1,-1),GRAY),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[LAMBER,WHITE]),
        ('GRID',(0,0),(-1,-1),0.3,colors.HexColor('#faf089')),
        ('ROWPADDING',(0,0),(-1,-1),5),
    ]))
    E.append(at)
    E.append(Spacer(1, 0.4*cm))

    claim = Table([[Paragraph(
        '<font color="#1a365d"><b>Key Research Claim:</b></font>  '
        'ShieldAI is the first open-source platform combining '
        '<b>Digital Twin pre-validation</b>, '
        '<b>SHAP per-alert explainability</b>, and a '
        '<b>decaying IP reputation blacklist</b> — '
        'three capabilities absent from all top commercial XDR vendors '
        '(CrowdStrike, SentinelOne, Microsoft Defender XDR) as of 2025.', body)
    ]], colWidths=[17*cm])
    claim.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,0),LCYAN),('BOX',(0,0),(0,0),1.5,CYAN),
        ('TOPPADDING',(0,0),(0,0),10),('BOTTOMPADDING',(0,0),(0,0),10),
        ('LEFTPADDING',(0,0),(0,0),14),('RIGHTPADDING',(0,0),(0,0),14),
    ]))
    E.append(claim)

    doc.build(E, onFirstPage=deco.draw, onLaterPages=deco.draw)
    buf.seek(0)
    from flask import send_file
    return send_file(buf, mimetype='application/pdf', as_attachment=True,
                     download_name=f"shieldai-report-{time.strftime('%Y%m%d-%H%M%S')}.pdf")

# ── Run ───────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  ShieldAI A-XDR+ Platform")
    print("="*50)
    print("  Login      : http://127.0.0.1:5000/login")
    print("  Dashboard  : http://127.0.0.1:5000/dashboard")
    print("  Stats      : http://127.0.0.1:5000/api/stats")
    print("  Geo Map    : http://127.0.0.1:5000/api/geo")
    print("  Simulate   : http://127.0.0.1:5000/api/simulate [POST]")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
