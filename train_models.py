import pandas as pd
import numpy as np
import joblib, os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize

print("=" * 60)
print("  ShieldAI A-XDR+ — Model Training & Evaluation")
print("=" * 60)

# Load data
print("\n[1/7] Loading NSL-KDD dataset...")
train = pd.read_csv('data/KDDTrain+.csv')
test  = pd.read_csv('data/KDDTest+.csv')

X_train = train.drop('label', axis=1)
y_train = train['label']
X_test  = test.drop('label', axis=1)
y_test  = test['label']

print(f"      Train samples : {len(train):,}")
print(f"      Test  samples : {len(test):,}")
print(f"      Features      : {X_train.shape[1]}")

# Encode labels
print("\n[2/7] Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
classes     = list(le.classes_)
print(f"      Classes: {classes}")

# Scale features
print("\n[3/7] Scaling features (StandardScaler)...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Random Forest
print("\n[4/7] Training Random Forest (n_estimators=100, n_jobs=-1)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_enc)

# Evaluate
print("\n[5/7] Evaluating — computing all IEEE metrics...")
y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)

accuracy  = accuracy_score(y_test_enc, y_pred)
precision = precision_score(y_test_enc, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test_enc, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)

precision_per = precision_score(y_test_enc, y_pred, average=None, zero_division=0)
recall_per    = recall_score(y_test_enc, y_pred, average=None, zero_division=0)
f1_per        = f1_score(y_test_enc, y_pred, average=None, zero_division=0)

y_test_bin = label_binarize(y_test_enc, classes=range(len(classes)))
auc        = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')

normal_idx  = classes.index('Normal')
normal_mask = (y_test_enc == normal_idx)
fpr_val     = np.sum((y_pred != normal_idx) & normal_mask) / max(np.sum(normal_mask), 1)

cm = confusion_matrix(y_test_enc, y_pred)

# Print Results
print(f"\n{'='*60}")
print(f"  PERFORMANCE METRICS  (IEEE Format)")
print(f"{'='*60}")
print(f"  Accuracy   : {accuracy  * 100:.2f}%")
print(f"  Precision  : {precision * 100:.2f}%  (weighted avg)")
print(f"  Recall     : {recall    * 100:.2f}%  (weighted avg)")
print(f"  F1-Score   : {f1        * 100:.2f}%  (weighted avg)")
print(f"  AUC-ROC    : {auc:.4f}  (macro OvR)")
print(f"  FPR        : {fpr_val   * 100:.2f}%  (Normal misclassified)")
print(f"{'='*60}")

print("\n  Per-Class Breakdown:")
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(f"  {'-'*46}")
for i, cls in enumerate(classes):
    print(f"  {cls:<12} {precision_per[i]*100:>9.2f}% {recall_per[i]*100:>9.2f}% {f1_per[i]*100:>9.2f}%")

print(f"\n  Full Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=classes))

print("\n  Confusion Matrix (rows=actual, cols=predicted):")
header = "           " + "".join(f"{c:>10}" for c in classes)
print(header)
for i, cls in enumerate(classes):
    row = f"  {cls:<10}" + "".join(f"{cm[i][j]:>10}" for j in range(len(classes)))
    print(row)

# Comparison Table
print(f"\n{'='*60}")
print("  COMPARISON WITH EXISTING SYSTEMS")
print(f"{'='*60}")
comparison = [
    ("Snort (Traditional IDS)",   78.5,  False, False, "Signature-based only"),
    ("Suricata IDS",               80.2,  False, False, "Signature + basic rules"),
    ("KDD-RF Baseline (2019)",     89.4,  False, False, "RF, no XAI"),
    ("CNN-IDS (Yin et al. 2022)", 91.3,  False, False, "CNN, no explainability"),
    ("LSTM-IDS (Liu et al. 2023)",93.7,  False, False, "LSTM only, no response"),
    ("ShieldAI A-XDR+ (Ours)",   accuracy * 100, True, True, "RF+LSTM+SHAP+AutoResponse"),
]
print(f"  {'System':<32} {'Accuracy':>9} {'XAI':>5} {'Auto-Response':>14}")
print(f"  {'-'*65}")
for name, acc, xai, resp, note in comparison:
    xai_s  = "YES" if xai  else "NO"
    resp_s = "YES" if resp else "NO"
    marker = " <-- OUR SYSTEM" if "Ours" in name else ""
    print(f"  {name:<32} {acc:>8.1f}%   {xai_s:<5}   {resp_s:<14}{marker}")
print(f"{'='*60}")

# Save models
print("\n[6/7] Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf,     'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le,     'models/label_encoder.pkl')

# Save enriched metadata
print("[7/7] Saving enriched metadata...")

per_class_metrics = {}
for i, cls in enumerate(classes):
    per_class_metrics[cls] = {
        "precision": round(float(precision_per[i]) * 100, 2),
        "recall":    round(float(recall_per[i])    * 100, 2),
        "f1":        round(float(f1_per[i])        * 100, 2),
    }

meta = {
    "accuracy":  round(accuracy  * 100, 2),
    "precision": round(precision * 100, 2),
    "recall":    round(recall    * 100, 2),
    "f1":        round(f1        * 100, 2),
    "auc_roc":   round(auc, 4),
    "fpr":       round(fpr_val   * 100, 2),
    "per_class": per_class_metrics,
    "confusion_matrix": cm.tolist(),
    "classes":   classes,
    "features":  list(X_train.columns),
    "n_train":   len(train),
    "n_test":    len(test),
    "comparison": [
        {"system": name, "accuracy": round(acc, 1), "xai": xai, "auto_response": resp}
        for name, acc, xai, resp, _ in comparison
    ],
}

with open('models/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n  Files saved: random_forest.pkl, scaler.pkl, label_encoder.pkl, metadata.json")
print(f"\n{'='*60}")
print("  Training COMPLETE — all IEEE metrics saved!")
print(f"{'='*60}")
