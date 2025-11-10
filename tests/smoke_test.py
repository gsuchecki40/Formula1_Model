#!/usr/bin/env python3
"""Simple smoke test: attempt to load the saved model and run one transform+predict.

Exits 0 on success or if model is missing (skipped). Exits non-zero on errors.
"""
import os
import sys
import joblib
import numpy as np


MODEL_PATH = os.path.join("data", "streamlit_model.joblib")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not present at {MODEL_PATH} — skipping smoke test.")
        return 0

    print("Loading model...")
    m = joblib.load(MODEL_PATH)
    if isinstance(m, dict):
        pre = m.get("preprocessor")
        booster = m.get("model")
    else:
        print("Model object not in expected dict format — cannot run smoke test")
        return 2

    if pre is None or booster is None:
        print("Preprocessor or model missing in saved object")
        return 2

    # build a single-row input using the preprocessor.feature_names_in_
    try:
        required = list(pre.feature_names_in_)
    except Exception as e:
        print("Preprocessor does not expose feature_names_in_", e)
        return 2

    import pandas as pd
    X = pd.DataFrame([{c: 0 if any(k in c.lower() for k in ['temp','grid','points','time','lap','season','round']) else 'OTHER' for c in required}])

    print("Transforming...")
    Xt = pre.transform(X)
    print("Transformed shape:", getattr(Xt, 'shape', None))

    try:
        import xgboost as xgb
    except Exception as e:
        print("xgboost not installed; cannot run booster prediction", e)
        return 0

    # Infer k from booster dump like the app does
    dump = booster.get_dump(dump_format='json')
    max_idx = -1
    import json, re
    for tree in dump:
        obj = json.loads(tree)
        stack = [obj]
        while stack:
            node = stack.pop()
            if isinstance(node, dict) and 'split' in node:
                s = node['split']
                m = re.match(r'f(\d+)', s)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx:
                        max_idx = idx
            elif isinstance(node, dict):
                for v in node.values():
                    stack.append(v)
            elif isinstance(node, list):
                for it in node:
                    stack.append(it)

    k = max_idx + 1 if max_idx >= 0 else None
    if k is not None and Xt.shape[1] >= k:
        Xt_small = Xt[:, :k]
    else:
        Xt_small = Xt

    print("Running booster.predict...")
    dmat = xgb.DMatrix(Xt_small)
    raw = booster.predict(dmat)
    print("Raw preds shape:", getattr(raw, 'shape', None))
    return 0


if __name__ == '__main__':
    sys.exit(main())
