"""
metrics_api.py — IEEE-grade metrics endpoints for ShieldAI A-XDR+
Endpoints: /api/metrics, /api/confusion-matrix, /api/comparison,
           /api/roc-curves, /api/class-distribution
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
        "per_class":     meta.get("per_class",     {}),
        "lstm_accuracy": meta.get("lstm_accuracy", 0),
        "dataset": {
            "name":     "NSL-KDD",
            "n_train":  meta.get("n_train",  0),
            "n_test":   meta.get("n_test",   0),
            "classes":  meta.get("classes",  []),
            "features": len(meta.get("features", [])),
        }
    })

@metrics_bp.route('/api/confusion-matrix')
def get_confusion_matrix():
    meta = load_meta()
    return jsonify({
        "matrix":  meta.get("confusion_matrix", []),
        "classes": meta.get("classes", []),
    })

@metrics_bp.route('/api/comparison')
def get_comparison():
    meta = load_meta()
    return jsonify(meta.get("comparison", []))

@metrics_bp.route('/api/roc-curves')
def get_roc_curves():
    """Per-class ROC curve data for dashboard chart."""
    meta = load_meta()
    return jsonify(meta.get("roc_curves", {}))

@metrics_bp.route('/api/class-distribution')
def get_class_distribution():
    """NSL-KDD class distribution for training set bar chart."""
    meta = load_meta()
    return jsonify({
        "distribution": meta.get("class_distribution", {}),
        "n_train": meta.get("n_train", 0),
    })
