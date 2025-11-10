import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np


try:
    import xgboost as xgb
except Exception:
    xgb = None

# Page config and light CSS
st.set_page_config(page_title="F1 Predictor", layout="wide")
_STYLE = """
<style>
    /* General spacing & font */
    html, body, [class*="css"], .stApp {
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    .stButton>button { height: 2.6rem; background:#0f62fe; color:white; border-radius:6px; }
    .stDownloadButton>button { background:#0f62fe; color:white; border-radius:6px }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        padding: .45rem; border-radius:6px; border:1px solid #e6ecf8;
    }
    /* Result table visuals */
    table.f1-table { border-collapse: collapse; width: 100%; font-family: Inter, Arial, sans-serif }
    table.f1-table thead tr { background: #0f62fe; color: #fff }
    table.f1-table th, table.f1-table td { padding: 10px; border-bottom: 1px solid #eef3ff; text-align: left }
    table.f1-table tbody tr:hover { background: #fbfdff }
    .result-card { background: linear-gradient(180deg,#ffffff,#fbfdff); padding:10px; border-radius:8px; box-shadow: 0 1px 6px rgba(20,36,66,0.08) }
    .app-header { display:flex; align-items:center; gap:12px }
    .logo-circle { width:44px; height:44px; border-radius:8px; background:#0f62fe; display:inline-block }
    .small-muted { color:#6b7280; font-size:0.95rem }
</style>
"""
st.markdown(_STYLE, unsafe_allow_html=True)


def get_model_path():
    # model is expected to be in the data folder next to this script
    base = os.path.dirname(__file__)
    return os.path.join(base, "data", "streamlit_model.joblib")


def extract_categorical_options(preprocessor):
    """Return a dict mapping categorical column name -> list of allowed categories (if available).

    Attempts to inspect ColumnTransformer / OneHotEncoder used in the preprocessor.
    """
    opts = {}
    if preprocessor is None:
        return opts
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        ct = None
        if isinstance(preprocessor, ColumnTransformer):
            ct = preprocessor
        elif hasattr(preprocessor, "named_steps"):
            for step in preprocessor.named_steps.values():
                if isinstance(step, ColumnTransformer):
                    ct = step
                    break

        if ct is None:
            return opts

        for name, transformer, cols in ct.transformers_:
            if cols is None:
                continue
            # Only consider string column names
            if not (isinstance(cols, (list, tuple)) and cols and all(isinstance(c, str) for c in cols)):
                continue

            # If transformer is a pipeline, try to find OneHotEncoder inside
            enc = None
            if hasattr(transformer, "named_steps"):
                for sub in transformer.named_steps.values():
                    if isinstance(sub, OneHotEncoder) or hasattr(sub, "categories_"):
                        enc = sub
                        break
            elif isinstance(transformer, OneHotEncoder) or hasattr(transformer, "categories_"):
                enc = transformer

            if enc is not None and hasattr(enc, "categories_"):
                cats = list(enc.categories_)
                # categories_ is a list aligned to input columns
                for col_name, cat_list in zip(cols, cats):
                    opts[col_name] = list(cat_list)
    except Exception:
        return opts
    return opts


def infer_feature_name(model):
    # Try a few heuristics to determine which feature name corresponds to grid position
    candidates = []
    try:
        if hasattr(model, "feature_names_in_"):
            candidates = list(model.feature_names_in_)
    except Exception:
        candidates = []

    # If pipeline, try to inspect last estimator
    if not candidates:
        try:
            # many sklearn Pipelines expose named_steps
            if hasattr(model, "named_steps"):
                last = list(model.named_steps.values())[-1]
                if hasattr(last, "feature_names_in_"):
                    candidates = list(last.feature_names_in_)
        except Exception:
            candidates = candidates

    # normalize and pick a column name that contains 'grid'
    for c in candidates:
        if "grid" in c.lower():
            return c

    # fallback names commonly used
    for c in ["grid_position", "gridpos", "grid", "Grid", "GridPosition"]:
        if c in candidates:
            return c

    return None


def build_feature_dataframe(user_df, model, race_inputs=None):
    """Attempt to construct a feature DataFrame that the model can consume.
    Strategy:
    - If model advertises feature_names_in_, attempt to create those cols using Grid and Driver as needed.
    - Otherwise, pass a single-column DF with the grid positions.
    The function returns (X, info_string) where X is the DataFrame to pass to model.predict
    """
    info = []
    # If model is a dict with a preprocessor, try to use its expected input names
    required_cols = None
    if isinstance(model, dict) and model.get("preprocessor") is not None:
        pre = model.get("preprocessor")
        if hasattr(pre, "feature_names_in_") and pre.feature_names_in_ is not None:
            required_cols = list(pre.feature_names_in_)

    # If no required cols discovered, try heuristics
    if required_cols is None:
        feature_name = infer_feature_name(model)
        if feature_name:
            X = pd.DataFrame()
            X[feature_name] = user_df["Grid"].astype(int)
            info.append(f"Mapped grid positions to feature '{feature_name}'.")
            return X, " ".join(info)

        X = pd.DataFrame()
        X["Grid"] = user_df["Grid"].astype(int)
        info.append("Model feature names not detected; passing single column 'Grid'.")
        return X, " ".join(info)

    # Build DataFrame with required columns
    n = len(user_df)
    X = pd.DataFrame(index=range(n), columns=required_cols)

    # Fill driver-level fields from user_df when possible
    # Map common names
    if "Driver" in required_cols and "Driver" in user_df.columns:
        X["Driver"] = user_df["Driver"].astype(str)

    # Grid mapping: look for a required column that contains 'grid' (case-insensitive)
    grid_col = None
    for c in required_cols:
        if "grid" in c.lower():
            grid_col = c
            break
    if grid_col is not None:
        X[grid_col] = user_df["Grid"].astype(float)

        # Race-level numeric defaults: prefer values in user_df if present, otherwise use reasonable defaults
        race_numeric_defaults = {
            "AirTemp_C": 20.0,
            "TrackTemp_C": 25.0,
            "Humidity_%": 50.0,
            "Pressure_hPa": 1013.25,
            "WindSpeed_mps": 1.0,
            "WindDirection_deg": 0.0,
            "Rain": 0.0,
            "Season": 2025,
            "Round": 1,
        }
        # override defaults with provided race_inputs
        if isinstance(race_inputs, dict):
            for k, v in race_inputs.items():
                race_numeric_defaults[k] = v

        for col, default in race_numeric_defaults.items():
            if col in required_cols:
                X[col] = default

    # Determine numeric vs categorical expected input columns.
    numeric_cols = set()
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        pre = None
        if isinstance(model, dict):
            pre = model.get("preprocessor")
        # find the ColumnTransformer inside preprocessor (it may be wrapped in a Pipeline)
        ct = None
        try:
            from sklearn.compose import ColumnTransformer
            if isinstance(pre, ColumnTransformer):
                ct = pre
            elif hasattr(pre, "named_steps"):
                for step in pre.named_steps.values():
                    if isinstance(step, ColumnTransformer):
                        ct = step
                        break
        except Exception:
            ct = None

        if ct is not None:
            for tname, transformer, cols in ct.transformers_:
                if not cols:
                    continue
                is_numeric_transformer = False
                # If the transformer name indicates numeric, mark it
                if isinstance(tname, str) and "num" in tname.lower():
                    is_numeric_transformer = True

                # If the transformer or its pipeline steps include a numeric imputer/scaler, mark it
                try:
                    if hasattr(transformer, "named_steps"):
                        for sub in transformer.named_steps.values():
                            if isinstance(sub, (SimpleImputer, StandardScaler)):
                                is_numeric_transformer = True
                                break
                            # detect SimpleImputer by attribute
                            if hasattr(sub, "strategy") and getattr(sub, "strategy", None) in ("median", "mean"):
                                is_numeric_transformer = True
                                break
                    else:
                        if isinstance(transformer, (SimpleImputer, StandardScaler)):
                            is_numeric_transformer = True
                        elif hasattr(transformer, "strategy") and getattr(transformer, "strategy", None) in ("median", "mean"):
                            is_numeric_transformer = True
                except Exception:
                    pass

                if is_numeric_transformer and isinstance(cols, (list, tuple)):
                    for cc in cols:
                        numeric_cols.add(cc)
    except Exception:
        numeric_cols = set()

    # If no numeric_cols discovered, fall back to keyword-based inference
    if not numeric_cols:
        numeric_keywords = ["temp", "time", "parsed", "points", "avg", "grid", "number", "lap", "laps", "pressure", "wind", "humidity", "season", "round", "q1", "q2", "q3", "delta", "deviation"]
        for c in required_cols:
            if any(k in c.lower() for k in numeric_keywords):
                numeric_cols.add(c)

    # If user provided additional columns in the editable table, copy them
    for c in user_df.columns:
        if c in required_cols:
            X[c] = user_df[c]

    # Fill remaining required columns with sensible defaults depending on numeric/categorical
    for col in required_cols:
        if col in X and pd.isna(X[col]).all():
            if col in numeric_cols:
                X[col] = 0
            else:
                X[col] = "OTHER"

    # Fill categorical defaults
    categorical_defaults = {"TeamName": "OTHER", "TeamId": "other", "ClassifiedPosition": "OTHER", "Time": "missing", "Status": "OTHER"}
    for col, default in categorical_defaults.items():
        if col in required_cols:
            X[col] = default

    # Ensure numeric columns are numeric where possible
    for c in X.columns:
        if X[c].dtype == object:
            # try convert to numeric where it looks numeric
            try:
                X[c] = pd.to_numeric(X[c])
            except Exception:
                pass

    info.append(f"Built DataFrame with required columns ({len(required_cols)}). Missing values may be filled with defaults.")
    return X, " ".join(info)


def main():
    # nicer header
    st.markdown(_STYLE, unsafe_allow_html=True)
    with st.container():
        c1, c2 = st.columns([0.12, 1])
        with c1:
            st.markdown("<div class='logo-circle'></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='app-header'><div><h2 style='margin:0'>Formula 1 — Predicted Finishing Order</h2><div class='small-muted'>Enter drivers, grid positions and race conditions, then click Predict</div></div></div>", unsafe_allow_html=True)
    st.write("")

    model_path = get_model_path()
    model = None
    preprocessor = None
    booster = None
    calibrator = None
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("Model loaded from data/streamlit_model.joblib")
            if isinstance(model, dict):
                preprocessor = model.get('preprocessor')
                booster = model.get('model')
                calibrator = model.get('calibrator')
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()
    else:
        st.error(f"Model not found at {model_path}. Please ensure `data/streamlit_model.joblib` exists.")
        st.stop()

    n = st.number_input("Number of drivers", min_value=1, max_value=30, value=20, step=1)

    # race-level inputs (weather, session info) — these are shared across all drivers
    st.sidebar.header("Race / Session inputs")
    season = st.sidebar.number_input("Season", value=2025)
    round_ = st.sidebar.number_input("Round", value=1)
    air_temp = st.sidebar.number_input("Air Temp (C)", value=20.0, format="%.1f")
    track_temp = st.sidebar.number_input("Track Temp (C)", value=25.0, format="%.1f")
    humidity = st.sidebar.number_input("Humidity (%)", value=50.0, format="%.1f")
    pressure = st.sidebar.number_input("Pressure (hPa)", value=1013.25, format="%.2f")
    wind_speed = st.sidebar.number_input("Wind Speed (m/s)", value=1.0, format="%.2f")
    wind_dir = st.sidebar.number_input("Wind Direction (deg)", value=0.0, format="%.0f")
    rain = st.sidebar.selectbox("Rain?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    # Instead of a raw table, use a per-driver interactive form stored in session_state
    # Extract categorical options (TeamName, Driver, etc.) from preprocessor if present
    cat_options = extract_categorical_options(preprocessor)

    # Initialize drivers in session_state if needed or if n changed
    if "drivers" not in st.session_state or st.session_state.get("drivers_n") != n:
        default_team = cat_options.get("TeamName", ["OTHER"])[0] if cat_options.get("TeamName") else "OTHER"
        st.session_state["drivers"] = [
            {"Driver": f"Driver {i+1}", "TeamName": default_team, "Grid": i + 1} for i in range(n)
        ]
        st.session_state["drivers_n"] = n

    st.markdown("Edit driver details below. Use the Add / Remove buttons to modify the list.")

    # Controls to add/remove drivers
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Add driver"):
            idx = len(st.session_state["drivers"]) + 1
            default_team = cat_options.get("TeamName", ["OTHER"])[0] if cat_options.get("TeamName") else "OTHER"
            st.session_state["drivers"].append({"Driver": f"Driver {idx}", "TeamName": default_team, "Grid": idx})
    with cols[1]:
        if st.button("Remove last"):
            if st.session_state["drivers"]:
                st.session_state["drivers"].pop()

    # Render per-driver expanders with inputs
    for i, drv in enumerate(st.session_state["drivers"]):
        key_prefix = f"drv_{i}"
        # Each driver expander contains controls and extra optional fields
        with st.expander(f"#{i+1} — {drv.get('Driver','Driver')}", expanded=False):
            # top row: name, team, grid, move up/down, delete
            r1 = st.columns([3, 3, 1, 0.5, 0.5, 1])
            drv_name = r1[0].text_input("Driver name", value=drv.get("Driver", f"Driver {i+1}"), key=key_prefix + "_name")

            team_opts = cat_options.get("TeamName", None)
            if team_opts:
                # keep OTHER option at the end
                opts = team_opts + [t for t in ["OTHER"] if t not in team_opts]
                default_idx = opts.index(drv.get("TeamName")) if drv.get("TeamName") in opts else len(opts) - 1
                drv_team = r1[1].selectbox("Team", options=opts, index=default_idx, key=key_prefix + "_team")
            else:
                drv_team = r1[1].text_input("Team", value=drv.get("TeamName", "OTHER"), key=key_prefix + "_team")

            grid_val = r1[2].number_input("Grid", min_value=1, max_value=30, value=int(drv.get("Grid", i + 1)), key=key_prefix + "_grid")

            # move up / down / delete
            if r1[3].button("↑", key=key_prefix + "_up"):
                if i > 0:
                    st.session_state["drivers"][i - 1], st.session_state["drivers"][i] = st.session_state["drivers"][i], st.session_state["drivers"][i - 1]
                    st.experimental_rerun()
            if r1[4].button("↓", key=key_prefix + "_down"):
                if i < len(st.session_state["drivers"]) - 1:
                    st.session_state["drivers"][i + 1], st.session_state["drivers"][i] = st.session_state["drivers"][i], st.session_state["drivers"][i + 1]
                    st.experimental_rerun()
            if r1[5].button("Delete", key=key_prefix + "_del"):
                st.session_state["drivers"].pop(i)
                st.experimental_rerun()

            # second row: optional numeric fields
            r2 = st.columns([1.5, 1.5, 1.5, 2])
            driver_number = r2[0].number_input("DriverNumber", min_value=0, max_value=999, value=int(drv.get("DriverNumber", 0)), key=key_prefix + "_num")
            avg_quali = r2[1].number_input("AvgQualiTime (s)", value=float(drv.get("AvgQualiTime", 0.0)), format="%.3f", key=key_prefix + "_quali")
            points = r2[2].number_input("Points", value=float(drv.get("Points", 0.0)), format="%.2f", key=key_prefix + "_points")

            # ClassifiedPosition (categorical) - use preprocessor categories if present
            class_opts = cat_options.get("ClassifiedPosition", None)
            if class_opts:
                classified = r2[3].selectbox("ClassifiedPosition", options=class_opts + ["OTHER"], index=(class_opts.index(drv.get("ClassifiedPosition")) if drv.get("ClassifiedPosition") in class_opts else len(class_opts)), key=key_prefix + "_class")
            else:
                classified = r2[3].text_input("ClassifiedPosition", value=drv.get("ClassifiedPosition", "OTHER"), key=key_prefix + "_class")

            # persist back to session_state
            st.session_state["drivers"][i]["Driver"] = drv_name
            st.session_state["drivers"][i]["TeamName"] = drv_team
            st.session_state["drivers"][i]["Grid"] = int(grid_val)
            st.session_state["drivers"][i]["DriverNumber"] = int(driver_number)
            st.session_state["drivers"][i]["AvgQualiTime"] = float(avg_quali)
            st.session_state["drivers"][i]["Points"] = float(points)
            st.session_state["drivers"][i]["ClassifiedPosition"] = classified

    # Build editable DataFrame from session_state
    edited = pd.DataFrame(st.session_state["drivers"])

    # Validation: duplicate grid positions
    if edited["Grid"].duplicated().any():
        st.warning("Duplicate grid positions detected — predicted order may be incorrect. Consider ensuring unique grid values.")

    # Sidebar: show model feature info and categorical options
    with st.sidebar.expander("Model feature info", expanded=False):
        if preprocessor is not None and hasattr(preprocessor, 'feature_names_in_'):
            st.markdown(f"**Preprocessor expects {len(preprocessor.feature_names_in_)} input columns**")
            st.text_area("Feature names (copyable)", value=", ".join(list(preprocessor.feature_names_in_)), height=120)
        else:
            st.write("Preprocessor feature names not available.")

        if cat_options:
            st.markdown("**Categorical options (sample)**")
            for k, v in cat_options.items():
                st.write(f"- {k}: {v[:20]}")
        else:
            st.write("No categorical options detected in preprocessor.")

    if st.button("Predict"):
        # basic validation
        if "Driver" not in edited.columns or "Grid" not in edited.columns:
            st.error("Table must contain 'Driver' and 'Grid' columns.")
        else:
            try:
                # include race-level values in a small dict for defaults
                race_inputs = {
                    'Season': season,
                    'Round': round_,
                    'AirTemp_C': air_temp,
                    'TrackTemp_C': track_temp,
                    'Humidity_%': humidity,
                    'Pressure_hPa': pressure,
                    'WindSpeed_mps': wind_speed,
                    'WindDirection_deg': wind_dir,
                    'Rain': rain,
                }

                X, info = build_feature_dataframe(edited, model, race_inputs)
                if info:
                    st.info(info)
                # Depending on the saved object shape, handle predictions differently
                preds = None
                if isinstance(model, dict):
                    pre = model.get('preprocessor')
                    booster = model.get('model')
                    calibrator = model.get('calibrator')

                    # transform through preprocessor
                    try:
                        X_trans = pre.transform(X)
                    except Exception as e:
                        st.error(f"Preprocessor transform failed: {e}")
                        st.stop()

                    # ensure xgboost available
                    if xgb is None:
                        st.error("xgboost is required to run this model. Install xgboost and restart the app.")
                        st.stop()

                    try:
                        # xgboost Booster may expect a different number of features than the preprocessor currently outputs
                        # try to infer expected feature count from the model dump (f0..fN)
                        expected_k = None
                        try:
                            dump = booster.get_dump(dump_format='json')
                            import json, re
                            max_idx = -1
                            for tree in dump:
                                obj = json.loads(tree)
                                stack = [obj]
                                while stack:
                                    node = stack.pop()
                                    if isinstance(node, dict):
                                        if 'split' in node:
                                            s = node['split']
                                            m = re.match(r'f(\d+)', s)
                                            if m:
                                                idx = int(m.group(1))
                                                if idx > max_idx:
                                                    max_idx = idx
                                        for v in node.values():
                                            stack.append(v)
                                    elif isinstance(node, list):
                                        for it in node:
                                            stack.append(it)
                            if max_idx >= 0:
                                expected_k = max_idx + 1
                        except Exception:
                            expected_k = None

                        # If expected_k is known and doesn't match, try slicing or padding
                        if expected_k is not None:
                            if hasattr(X_trans, 'shape') and X_trans.shape[1] > expected_k:
                                st.info(f"Preprocessor produced {X_trans.shape[1]} features but booster uses {expected_k}; slicing to first {expected_k} columns.")
                                X_for_booster = X_trans[:, :expected_k]
                            elif hasattr(X_trans, 'shape') and X_trans.shape[1] < expected_k:
                                # pad with zeros
                                import numpy as _np
                                pad_cols = expected_k - X_trans.shape[1]
                                st.info(f"Preprocessor produced {X_trans.shape[1]} features but booster expects {expected_k}; padding with {pad_cols} zero columns.")
                                X_for_booster = _np.hstack([X_trans, _np.zeros((X_trans.shape[0], pad_cols))])
                            else:
                                X_for_booster = X_trans
                        else:
                            X_for_booster = X_trans

                        dmat = xgb.DMatrix(X_for_booster)
                        raw_preds = booster.predict(dmat)
                    except Exception as e:
                        st.error(f"Booster prediction failed: {e}")
                        st.stop()

                    # apply calibrator if present
                    if calibrator is not None:
                        try:
                            # calibrator expects shape (n_samples, 1)
                            preds = calibrator.predict(raw_preds.reshape(-1, 1))
                        except Exception as e:
                            st.warning(f"Calibrator failed: {e} — returning raw predictions")
                            preds = raw_preds
                    else:
                        preds = raw_preds

                else:
                    # legacy single-estimator case
                    try:
                        preds = model.predict(X)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                # attach predictions and sort
                out = edited.copy()
                out["PredictedFinish"] = preds
                # If predicted finish is numeric where 1 is best, sort ascending
                try:
                    out_sorted = out.sort_values(by=["PredictedFinish"])  # ascending
                except Exception:
                    out_sorted = out

                out_sorted = out_sorted.reset_index(drop=True)

                st.success("Prediction complete. Showing predicted finishing order:")

                # Render a polished HTML table with top-3 highlight and light banding
                def _row_html(r):
                    rank = int(r['PredictedFinish'])
                    # use softer, less-saturated highlight colors
                    if rank <= 3:
                        bg = '#e6ffd9'  # very light green
                    elif rank <= 10:
                        bg = '#f4f6f8'  # subtle light gray-blue
                    else:
                        bg = 'white'
                    return (
                        f"<tr style=\"background:{bg};text-align:left;color:#111;\">"
                        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{int(r['PredictedFinish'])}</td>"
                        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{r.get('Driver','')}</td>"
                        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{r.get('TeamName','')}</td>"
                        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{r.get('Grid','')}</td>"
                        f"<td style='padding:8px;border-bottom:1px solid #ddd'>{float(r.get('PredictedFinish',0)):.4f}</td>"
                        f"</tr>"
                    )

                rows_html = '\n'.join([_row_html(row) for _, row in out_sorted.iterrows()])
                table_html = (
                    "<div style='overflow:auto'>"
                    "<table style='border-collapse:collapse;width:100%;font-family:Arial,Helvetica,sans-serif'>"
                    "<thead><tr style='background:#333;color:#fff;text-align:left'>"
                    "<th style='padding:10px'>Rank</th><th>Driver</th><th>Team</th><th>Grid</th><th>Score</th></tr></thead>"
                    f"<tbody>{rows_html}</tbody></table></div>"
                )

                st.markdown(table_html, unsafe_allow_html=True)

                # download
                csv = out_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv, file_name="predicted_order.csv", mime="text/csv")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
