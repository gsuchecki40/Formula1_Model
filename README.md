# Formula1 Model - Streamlit App

This small Streamlit app lets you load the saved model at `data/streamlit_model.joblib`, enter driver names next to their grid positions, and get a predicted finishing order.

How to run

1. Create a virtual environment and install dependencies. Example (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
# Use the 'python3 -m pip' form to ensure you call pip for the correct Python.
# On some systems the plain `pip` command is not present; `pip3` or `python3 -m pip` is more reliable.
python3 -m pip install -r requirements.txt
```

2. Run the Streamlit app from the repo root:

```bash
streamlit run app.py
```

XGBoost note

- The saved model requires the `xgboost` Python package. If you see an error like `ModuleNotFoundError: No module named 'xgboost'`, install it with one of the following approaches:

	- Using pip (recommended where wheels are available):

		```bash
		python3 -m pip install xgboost
		```

	- Using conda (recommended on macOS/Apple Silicon or if pip build fails):

		```bash
		conda install -c conda-forge xgboost
		```

	- Or just install all requirements (added `xgboost` to `requirements.txt`):

		```bash
		python3 -m pip install -r requirements.txt
		```

If `xgboost` fails to install with pip, try the conda-based command above.

Usage notes

- The app expects the model file at `data/streamlit_model.joblib` relative to the repo root. If the file is missing, the app will show an error.
- The UI shows an editable table with two columns: `Driver` and `Grid`. Edit driver names and grid positions as needed.
- When you click Predict the app will attempt to map your `Grid` values into whatever feature name the model expects (heuristic). If the model expects a different set of features (for example encoded driver IDs or additional columns), the app will attempt a simple fallback (single-column `Grid`) and may fail; in that case you'll need to adapt the code to construct the correct input DataFrame matching the model's training features.

Troubleshooting

- If you get an error when loading or predicting, open a Python REPL and inspect the model with:

```python
import joblib
m = joblib.load('data/streamlit_model.joblib')
print(type(m))
print(getattr(m, 'feature_names_in_', None))
```

Use that info to update `app.py` so the DataFrame columns match the model's expected features.

If you'd like, I can inspect the model file and refine the app to build the exact features the model needs. There is also a helper script at `scripts/inspect_model.py` that prints the top-level structure of the saved model to help with that:

```bash
python3 scripts/inspect_model.py
```

## Git: committing only necessary files

This repo may contain large or environment-specific files (virtualenvs, data, logs). To avoid accidentally pushing thousands of files, a `.gitignore` is included and you can follow these steps to commit only the important files:

```bash
# review unstaged files
git status --short

# add only project files (example)
git add app.py README.md requirements.txt .gitignore run_streamlit.sh scripts/inspect_model.py

# commit and push
git commit -m "Add Streamlit app and helpers"
git push origin main
```

If you already accidentally added large files (for example the virtualenv), remove them from the index but keep them locally with:

```bash
# stop tracking a file that's already committed
git rm --cached -r .venv
git commit -m "Remove venv from repo"
git push origin main
```

If you want, I can prepare a minimal commit for you (adding only the core files) — tell me and I'll run the git commands here.
