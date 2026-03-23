import pandas as pd
import numpy as np
import joblib, os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

print("="*55)
print("  ShieldAI — Model Training (Day 3)")
print("="*55)

# Load data
print("\n[1/6] Loading dataset...")
train = pd.read_csv('data/KDDTrain+.csv')
test  = pd.read_csv('data/KDDTest+.csv')

X_train = train.drop('label', axis=1)
y_train = train['label']
X_test  = test.drop('label', axis=1)
y_test  = test['label']

# Encode labels
print("[2/6] Encoding labels...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
print(f"      Classes: {list(le.classes_)}")

# Scale features
print("[3/6] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Random Forest
print("[4/6] Training Random Forest (100 trees)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train_enc)

# Evaluate
print("[5/6] Evaluating model...")
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test_enc, y_pred)

# AUC-ROC (multiclass)
y_test_bin = label_binarize(y_test_enc, classes=range(len(le.classes_)))
y_prob = rf.predict_proba(X_test_scaled)
auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')

# False Positive Rate
normal_idx = list(le.classes_).index('Normal')
normal_mask = (y_test_enc == normal_idx)
fpr = np.sum((y_pred != normal_idx) & normal_mask) / np.sum(normal_mask)

print(f"\n{'='*55}")
print(f"  RESULTS")
print(f"{'='*55}")
print(f"  Accuracy  : {accuracy*100:.2f}%")
print(f"  AUC-ROC   : {auc:.4f}")
print(f"  FPR       : {fpr*100:.2f}%")
print(f"{'='*55}")
print("\n  Per-class breakdown:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Save models
print("[6/6] Saving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf,     'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le,     'models/label_encoder.pkl')

# Save metadata
meta = {
    "accuracy": round(accuracy * 100, 2),
    "auc_roc":  round(auc, 4),
    "fpr":      round(fpr * 100, 2),
    "classes":  list(le.classes_),
    "features": list(X_train.columns),
    "n_train":  len(train),
    "n_test":   len(test)
}
with open('models/metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n  Files saved:")
print("    models/random_forest.pkl")
print("    models/scaler.pkl")
print("    models/label_encoder.pkl")
print("    models/metadata.json")
print(f"\n{'='*55}")
print("  Day 3 COMPLETE — real models trained!")
print("  Next: SHAP explainability (Day 4)")
print(f"{'='*55}")
