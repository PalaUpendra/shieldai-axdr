"""
metrics_api.py — IEEE-grade metrics endpoints for ShieldAI A-XDR+
Adds: /api/metrics, /api/confusion-matrix, /api/comparison
Register this blueprint in app.py:
    from metrics_api import metrics_bp
    app.register_blueprint(metrics_bp)
"""
from flask import Blueprint, jsonify
import json, os

metrics_bp = Blueprint('metrics', __name__)

def load_meta():
    path = os.path.join(os.path.dirname(__file__), 'models', 'metadata.json')
    with open(path) as f:
        return json.load(f)

@metrics_bp.route('/api/metrics')
def get_metrics():
    """Returns full IEEE performance metrics."""
    meta = load_meta()
    return jsonify({
        "overall": {
            "accuracy":  meta.get("accuracy",  0),
            "precision": meta.get("precision", 0),
            "recall":    meta.get("recall",    0),
            "f1":        meta.get("f1",        0),
            "auc_roc":   meta.get("auc_roc",   0),
            "fpr":       meta.get("fpr",       0),
        },
        "per_class":     meta.get("per_class",        {}),
        "lstm_accuracy": meta.get("lstm_accuracy",    0),
        "dataset": {
            "name":    "NSL-KDD",
            "n_train": meta.get("n_train", 0),
            "n_test":  meta.get("n_test",  0),
            "classes": meta.get("classes", []),
            "features": len(meta.get("features", [])),
        }
    })

@metrics_bp.route('/api/confusion-matrix')
def get_confusion_matrix():
    """Returns confusion matrix as 2-D list with class labels."""
    meta = load_meta()
    return jsonify({
        "matrix":  meta.get("confusion_matrix", []),
        "classes": meta.get("classes", []),
    })

@metrics_bp.route('/api/comparison')
def get_comparison():
    """Returns comparison table vs existing IDS/XDR systems."""
    meta = load_meta()
    return jsonify(meta.get("comparison", []))
