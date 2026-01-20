import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from io import BytesIO
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Exoplanet Habitability", layout="wide")

DEFAULT_API = "https://habitability-of-exoplanet.onrender.com"
#DEFAULT_API = "http://localhost:5000"

st.sidebar.title("Settings")
BASE_URL = st.sidebar.text_input("API base URL", value=DEFAULT_API)

st.title("Exoplanet Habitability Explorer 🔭")

tabs = st.tabs(["Predict", "Top Habitable", "Feature Importance", "Model & Sampling Comparisons", "Dataset"])

final_features = [
    "HSI",
    "planet_density",
    "pl_eqt",
    "pl_rade",
    "pl_bmasse",
    "st_teff",
    "star_luminosity",
    # star types are encoded as one-hot
    "star_type_M",
    "star_type_K",
    "star_type_G",
]

# --- Predict Tab ---
with tabs[0]:
    st.header("Predict Habitability")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            HSI = st.number_input("HSI", value=0.5, step=0.01, format="%.3f")
            planet_density = st.number_input("Planet Density", value=5.5, step=0.01)
            pl_eqt = st.number_input("Equilibrium Temp (K)", value=300.0, step=1.0)
        with col2:
            pl_rade = st.number_input("Planet Radius (R⊕)", value=1.0, step=0.01)
            pl_bmasse = st.number_input("Planet Mass (M⊕)", value=1.0, step=0.01)
            st_teff = st.number_input("Stellar Teff (K)", value=5800.0, step=1.0)
        with col3:
            star_luminosity = st.number_input("Star Luminosity (L☉)", value=1.0, step=0.01)
            star_type = st.selectbox("Star Type", options=["M", "K", "G"], index=2)

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "HSI": float(HSI),
            "planet_density": float(planet_density),
            "pl_eqt": float(pl_eqt),
            "pl_rade": float(pl_rade),
            "pl_bmasse": float(pl_bmasse),
            "st_teff": float(st_teff),
            "star_luminosity": float(star_luminosity),
            "star_type_M": 1 if star_type == "M" else 0,
            "star_type_K": 1 if star_type == "K" else 0,
            "star_type_G": 1 if star_type == "G" else 0,
        }

        # Basic validation before sending to API
        try:
            if any(pd.isna(v) for v in payload.values()):
                st.error("Please provide all inputs before predicting.")
            else:
                res = requests.post(f"{BASE_URL.rstrip('/')}/predict", json=payload, timeout=20)
                if res.status_code != 200:
                    detail = res.text
                    try:
                        detail_json = res.json()
                        detail = json.dumps(detail_json, indent=2)
                    except Exception:
                        pass
                    st.error(f"Server returned {res.status_code}: {detail}")
                else:
                    data = res.json()
                    prob = data.get("habitability_probability")
                    label = data.get("habitability_prediction")
                    filled = data.get("filled_defaults", [])
                    st.success(f"Prediction: {label} — Probability: {prob}")
                    if filled:
                        st.info(f"Missing inputs were filled with dataset medians for: {', '.join(filled)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Top Habitable Tab ---
with tabs[1]:
    st.header("Top Habitable Exoplanets")
    try:
        # Use API endpoint instead of local CSV
        df_top = pd.DataFrame()
        try:
            resp = requests.get(f"{BASE_URL.rstrip('/')}/top-habitable", timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type or resp.text.strip().startswith("["):
                df_top = pd.DataFrame(resp.json())
            else:
                # fallback if API returned CSV text
                from io import StringIO
                df_top = pd.read_csv(StringIO(resp.text))
        except Exception as e:
            st.error(f"Failed to load top habitable data: {e}")
            df_top = pd.DataFrame()

        if df_top.empty:
            st.warning("No top habitable data available.")
        else:
            st.subheader("Top Habitable Table")
            st.dataframe(df_top, use_container_width=True)

            # export buttons (CSV / Excel / HTML)
            csv_bytes = df_top.to_csv(index=False).encode("utf-8")
            excel_io = BytesIO()
            df_top.to_excel(excel_io, index=False)
            html_str = df_top.to_html(index=False)

            col_exp1, col_exp2, col_exp3 = st.columns(3)
            with col_exp1:
                st.download_button("Export CSV", data=csv_bytes, file_name="top_habitable.csv", mime="text/csv")
            with col_exp2:
                st.download_button("Export Excel", data=excel_io.getvalue(), file_name="top_habitable.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            with col_exp3:
                st.download_button("Export HTML", data=html_str, file_name="top_habitable.html", mime="text/html")

            # Plotly visualization: ensure fig_avg is used
            try:
                if "star_type" in df_top.columns and "HSI" in df_top.columns:
                    grouped = df_top.groupby("star_type", as_index=False).mean()
                    fig_avg = px.bar(grouped, x="star_type", y="HSI", title="Average HSI by Star Type")
                    st.plotly_chart(fig_avg, use_container_width=True)
                else:
                    if "HSI" in df_top.columns and "pl_rade" in df_top.columns:
                        fig_avg = px.scatter(df_top, x="pl_rade", y="HSI", color=df_top.columns[0] if len(df_top.columns) > 0 else None, title="HSI vs Planet Radius")
                        st.plotly_chart(fig_avg, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to render top-habitable plots: {e}")

    except FileNotFoundError:
        st.error("Top habitable data file not found.")
    except Exception as e:
        st.error(f"Failed to load top habitable data: {e}")

# --- Feature Importance ---
with tabs[2]:
    st.header("Feature Importance")
    fi_path = Path(__file__).resolve().parent.parent / "notebook" / "feature_importance_ranking.csv"

    if not fi_path.exists():
        st.error(f"Feature importance file not found: {fi_path}")
    else:
        try:
            df_feat = pd.read_csv(fi_path)
        except Exception as e:
            st.error(f"Could not read feature importance CSV: {e}")
            df_feat = pd.DataFrame()

        if df_feat.empty:
            st.warning("Feature importance file is empty.")
        else:
            feat_cols = [c for c in df_feat.columns if "feat" in c.lower() or "feature" in c.lower()]
            imp_cols = [c for c in df_feat.columns if "imp" in c.lower() or "importance" in c.lower() or "score" in c.lower()]

            y_col = feat_cols[0] if feat_cols else df_feat.columns[0]
            x_col = imp_cols[0] if imp_cols else (df_feat.columns[1] if len(df_feat.columns) > 1 else df_feat.columns[0])

            try:
                df_feat[x_col] = pd.to_numeric(df_feat[x_col], errors="coerce")
                df_feat = df_feat.sort_values(by=x_col, ascending=False).reset_index(drop=True)
            except Exception:
                df_feat = df_feat.reset_index(drop=True)

            st.subheader("Feature Importance Table")
            st.dataframe(df_feat, use_container_width=True)

            max_options = max(3, min(len(df_feat), 50))
            top_n = st.slider("Adjust the figure size", min_value=3, max_value=max_options, value=min(10, len(df_feat)))
            plot_df = df_feat.head(top_n)

            fig, ax = plt.subplots(figsize=(8, max(3, len(plot_df) * 0.5)))
            sns.barplot(data=plot_df, x=x_col, y=y_col, ax=ax, palette="viridis")
            ax.set_title("Feature Importance")
            ax.set_xlabel(x_col.replace("_", " ").title())
            ax.set_ylabel(y_col.replace("_", " ").title())
            plt.tight_layout()
            st.pyplot(fig)

# --- Model & Sampling Comparisons ---
with tabs[3]:
    st.header("Model & Sampling Comparisons")

    col1, col2 = st.columns(2)

    # -------- Model Comparison (from API) --------
    with col1:
        try:
            resp = requests.get(f"{BASE_URL.rstrip('/')}/model-comparisons", timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type or resp.text.strip().startswith("["):
                df_mc = pd.DataFrame(resp.json())
            else:
                from io import StringIO
                df_mc = pd.read_csv(StringIO(resp.text))
            if df_mc.empty:
                st.info("Model comparison data is empty.")
            else:
                st.subheader("Model Comparison (Summary)")
                st.dataframe(df_mc, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load model comparison data from API: {e}")

    # -------- Sampling Techniques (from API) --------
    with col2:
        try:
            resp = requests.get(f"{BASE_URL.rstrip('/')}/sampling-comparisons", timeout=20)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type or resp.text.strip().startswith("["):
                df_sc = pd.DataFrame(resp.json())
            else:
                from io import StringIO
                df_sc = pd.read_csv(StringIO(resp.text))
            if df_sc.empty:
                st.info("Sampling techniques data is empty.")
            else:
                st.subheader("Sampling Techniques (Summary)")
                st.dataframe(df_sc, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load sampling techniques data from API: {e}")

    # Star-Planet Parameter Correlations (use API exo-final)
    st.subheader("Star-Planet Parameter Correlations")
    try:
        resp = requests.get(f"{BASE_URL.rstrip('/')}/exo-final", timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type or resp.text.strip().startswith("["):
            df_data = pd.DataFrame(resp.json())
        else:
            from io import StringIO
            df_data = pd.read_csv(StringIO(resp.text))

        if df_data.empty:
            st.warning("Dataset is empty.")
        else:
            cols = [c for c in final_features if c in df_data.columns]
            if not cols:
                cols = df_data.select_dtypes(include="number").columns.tolist()
            corr = df_data[cols].corr()

            st.markdown("Correlation matrix (rounded)")
            st.dataframe(corr.round(3), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, max(4, len(cols) * 0.3)))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
            ax.set_title("Feature Correlation Heatmap")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to render correlation plot from API: {e}")

# --- Full dataset / extras ---
with tabs[4]:
    st.header("Dataset")
    try:
        resp = requests.get(f"{BASE_URL.rstrip('/')}/exo-final", timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type or resp.text.strip().startswith("["):
            df_full = pd.DataFrame(resp.json())
        else:
            from io import StringIO
            df_full = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        st.error(f"Failed to load dataset from API: {e}")
        df_full = pd.DataFrame()

    if df_full.empty:
        st.warning("Dataset not available.")
    else:
        st.subheader("Full Dataset")
        st.dataframe(df_full, use_container_width=True)

        csv_bytes = df_full.to_csv(index=False).encode("utf-8")
        excel_io = BytesIO()
        df_full.to_excel(excel_io, index=False)
        html_str = df_full.to_html(index=False)

        col1_exp, col2_exp, col3_exp = st.columns(3)
        with col1_exp:
            st.download_button("Export CSV", data=csv_bytes, file_name="exo_final.csv", mime="text/csv")
        with col2_exp:
            st.download_button("Export Excel", data=excel_io.getvalue(), file_name="exo_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col3_exp:
            st.download_button("Export HTML", data=html_str, file_name="exo_final.html", mime="text/html")

st.markdown("---")
st.caption("Tip: run the Flask API (`python notebook/api/app.py`) and then run this Streamlit app (`streamlit run frontend/streamlit_app.py`).")
