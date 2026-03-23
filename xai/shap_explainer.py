import joblib, json
import numpy as np
import pandas as pd
import shap

print("="*55)
print("  ShieldAI — SHAP Explainability (Day 4)")
print("="*55)

# Load models
print("\n[1/4] Loading models...")
rf      = joblib.load('models/random_forest.pkl')
scaler  = joblib.load('models/scaler.pkl')
le      = joblib.load('models/label_encoder.pkl')
meta    = json.load(open('models/metadata.json'))
features = meta['features']

# Load a small sample
print("[2/4] Loading sample data...")
df = pd.read_csv('data/KDDTest+.csv')
X  = df.drop('label', axis=1).head(100)
X_scaled = scaler.transform(X)

# Build SHAP explainer
print("[3/4] Building SHAP explainer (takes ~30 seconds)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_scaled)

# Save explainer
import os
os.makedirs('xai', exist_ok=True)
joblib.dump(explainer, 'xai/shap_explainer.pkl')
print("      Saved: xai/shap_explainer.pkl")

# Show explanation for one sample
print("\n[4/4] Example explanation for sample #0:")
sample_idx   = 0
pred_class   = rf.predict(X_scaled[[sample_idx]])[0]
pred_label   = le.inverse_transform([pred_class])[0]
confidence   = rf.predict_proba(X_scaled[[sample_idx]])[0].max() * 100

# Top 5 features for this prediction
sv = shap_values[pred_class][sample_idx]
top5_idx = np.argsort(np.abs(sv))[::-1][:5]

print(f"\n{'='*55}")
print(f"  THREAT DETECTED: {pred_label}")
print(f"  Confidence      : {confidence:.1f}%")
print(f"{'='*55}")
print(f"  AI Reasoning (SHAP):")
for rank, i in enumerate(top5_idx, 1):
    bar_len = int(abs(sv[i]) / max(abs(sv)) * 20)
    bar = '█' * bar_len
    direction = '+' if sv[i] > 0 else '-'
    print(f"  {rank}. {features[i]:<30} {direction}{bar}")

print(f"\n  Natural Language:")
print(f"  The model flagged this as {pred_label} because")
print(f"  '{features[top5_idx[0]]}' was the strongest signal,")
print(f"  followed by '{features[top5_idx[1]]}' and")
print(f"  '{features[top5_idx[2]]}'.")
print(f"\n  Decision: BLOCK + LOG + ALERT")
print(f"{'='*55}")
print("  Day 4 COMPLETE — SHAP explainer ready!")
print("  Next: Flask API backend (Day 5)")
print(f"{'='*55}")
