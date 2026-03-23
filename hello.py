# ShieldAI — Setup Verification
# Run: python hello.py

print("=" * 50)
print("  AXDR PLATFORM FYP — Setup Check")
print("=" * 50)

import sys
print(f"\nPython version : {sys.version[:6]}")

import flask
print(f"Flask          : OK")

import sklearn
print(f"scikit-learn   : OK")

import numpy as np
print(f"NumPy          : OK")

import pandas as pd
print(f"Pandas         : OK")

try:
    import tensorflow as tf
    print(f"TensorFlow     : OK ({tf.__version__})")
except:
    print(f"TensorFlow     : run  pip install tensorflow-cpu")

try:
    import shap
    print(f"SHAP           : OK")
except:
    print(f"SHAP           : run  pip install shap")

print()
print("=" * 50)
print("  Day 1 COMPLETE — ready to build ShieldAI!")
print("  Next: download NSL-KDD dataset")
print("=" * 50)