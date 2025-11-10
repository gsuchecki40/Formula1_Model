"""Small helper to inspect the saved model in data/streamlit_model.joblib.

Run from the repo root:
    python3 scripts/inspect_model.py

This prints top-level keys and types so you can adapt the Streamlit app to the model's internal shape.
"""
import joblib
import os
import pprint


def main():
    path = os.path.join('data', 'streamlit_model.joblib')
    if not os.path.exists(path):
        print('Model file not found at', path)
        return

    m = joblib.load(path)
    print('Top-level model type:', type(m))
    if isinstance(m, dict):
        print('\nTop-level dict keys and types:')
        for k, v in m.items():
            print(f"- {k}: {type(v)}")
            # for small items, pretty print
            try:
                if isinstance(v, (list, tuple)) and len(v) <= 10:
                    pprint.pprint(v)
            except Exception:
                pass
    else:
        print('\nModel repr:')
        print(repr(m))


if __name__ == '__main__':
    main()
