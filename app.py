import io
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="EthicX MVP", layout="wide")


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def load_model(file):
    """Try pickle first, fall back to joblib."""
    raw = file.read()
    try:
        return pickle.loads(raw)
    except Exception:
        file.seek(0)
        return joblib.load(io.BytesIO(raw))


def get_predictions(model, X):
    preds = model.predict(X)
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = None
    return preds, proba

def group_rates(y_pred, sensitive, privileged, unprivileged):
    priv_mask = sensitive == privileged
    unpriv_mask = sensitive == unprivileged
    priv_rate = y_pred[priv_mask].mean() if priv_mask.sum() else np.nan
    unpriv_rate = y_pred[unpriv_mask].mean() if unpriv_mask.sum() else np.nan
    return priv_rate, unpriv_rate


def disparate_impact_ratio(y_pred, sensitive, privileged, unprivileged):
    priv_rate, unpriv_rate = group_rates(y_pred, sensitive, privileged, unprivileged)
    if priv_rate in (0, np.nan) or np.isnan(priv_rate):
        return np.nan
    return unpriv_rate / priv_rate


def statistical_parity_difference(y_pred, sensitive, privileged, unprivileged):
    priv_rate, unpriv_rate = group_rates(y_pred, sensitive, privileged, unprivileged)
    return unpriv_rate - priv_rate


def equal_opportunity_difference(y_true, y_pred, sensitive, privileged, unprivileged):
    priv_mask = (sensitive == privileged) & (y_true == 1)
    unpriv_mask = (sensitive == unprivileged) & (y_true == 1)
    priv_tpr = y_pred[priv_mask].mean() if priv_mask.sum() else np.nan
    unpriv_tpr = y_pred[unpriv_mask].mean() if unpriv_mask.sum() else np.nan
    return unpriv_tpr - priv_tpr


def accuracy_by_group(y_true, y_pred, sensitive):
    out = {}
    for g in sensitive.unique():
        mask = sensitive == g
        out[g] = accuracy_score(y_true[mask], y_pred[mask]) if mask.sum() else np.nan
    return out

st.sidebar.title("EthicX")
st.sidebar.caption("Explainability + Bias Audit — MVP")

data_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=["csv"])
model_file = st.sidebar.file_uploader("Upload trained model (.pkl / .joblib)", type=["pkl", "joblib"])
train_demo = st.sidebar.checkbox("No model? Train a quick RandomForest demo model instead", value=False)

st.title("⚖️ EthicX — Bias & Explainability Audit")
st.caption("Upload a dataset + a classifier to see SHAP, LIME, and fairness metrics side by side.")

if data_file is None:
    st.info("Upload a CSV dataset from the sidebar to get started.")
    st.stop()

df = load_csv(data_file)
st.subheader("1. Dataset preview")
st.dataframe(df.head())

cols = df.columns.tolist()
target_col = st.selectbox("Target column (what the model predicts)", cols, index=len(cols) - 1)
feature_cols = st.multiselect(
    "Feature columns (used by the model)",
    [c for c in cols if c != target_col],
    default=[c for c in cols if c != target_col],
)
sensitive_col = st.selectbox("Sensitive attribute (for bias audit)", [c for c in cols if c != target_col])

if not feature_cols:
    st.warning("Select at least one feature column.")
    st.stop()

X = df[feature_cols].copy()
y = df[target_col].copy()
sensitive = df[sensitive_col].copy()

X_encoded = pd.get_dummies(X, drop_first=False).astype(float)

model = None
if model_file is not None:
    try:
        model = load_model(model_file)
        st.success(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()
elif train_demo:
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    st.info("Trained a demo RandomForestClassifier on 75% of the uploaded data.")
else:
    st.warning("Upload a model file, or check 'train a quick demo model' in the sidebar.")
    st.stop()

try:
    X_encoded = X_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0)
    preds, proba = get_predictions(model, X_encoded)
except Exception as e:
    st.error(
        f"Model prediction failed on the encoded features ({e}). "
        "Make sure the uploaded model was trained on features matching this dataset's schema, "
        "or use the demo-train option."
    )
    st.stop()

st.subheader("2. Model performance")
col1, col2 = st.columns([1, 2])
with col1:
    try:
        acc = accuracy_score(y, preds)
        st.metric("Overall accuracy", f"{acc:.3f}")
    except Exception:
        st.write("Accuracy unavailable (check target encoding).")
with col2:
    try:
        cm = confusion_matrix(y, preds)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    except Exception:
        st.write("Confusion matrix unavailable.")

st.subheader("3. SHAP — global & local explainability")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_encoded)

    # Handle classifier outputs where TreeExplainer may return a list
    if isinstance(shap_values, list):
        # prefer the positive class if available (index 1), fall back to first
        if len(shap_values) > 1:
            shap_vals_for_plot = shap_values[1]
        else:
            shap_vals_for_plot = shap_values[0]
    else:
        shap_vals_for_plot = shap_values

    # If we received a raw numpy array, wrap it in a shap.Explanation
    if isinstance(shap_vals_for_plot, (list, tuple)) or isinstance(shap_vals_for_plot, np.ndarray):
        base = getattr(explainer, "expected_value", None)
        if isinstance(base, (list, tuple, np.ndarray)):
            base_val = base[1] if len(base) > 1 else base[0]
        else:
            base_val = base
        shap_vals_for_plot = shap.Explanation(
            values=np.array(shap_vals_for_plot),
            base_values=base_val,
            data=X_encoded,
            feature_names=X_encoded.columns.tolist(),
        )

    # If Explanation values are 3D (samples x features x classes), pick a single class
    vals = getattr(shap_vals_for_plot, 'values', None)
    if vals is not None and getattr(vals, 'ndim', None) == 3:
        class_idx = 1 if vals.shape[2] > 1 else 0
        base = getattr(shap_vals_for_plot, 'base_values', None)
        if isinstance(base, (list, tuple, np.ndarray)) and len(base) > class_idx:
            base_val = base[class_idx]
        else:
            base_val = base
        shap_vals_for_plot = shap.Explanation(
            values=vals[:, :, class_idx],
            base_values=base_val,
            data=X_encoded,
            feature_names=X_encoded.columns.tolist(),
        )

    st.markdown("**Global feature importance**")
    fig1 = plt.figure()
    shap.plots.bar(shap_vals_for_plot, show=False, max_display=10)
    st.pyplot(fig1)
    plt.clf()

    st.markdown("**Local explanation for a single row**")
    row_idx = st.number_input("Row index", 0, len(X_encoded) - 1, 0)
    fig2 = plt.figure()
    # For local waterfall, index into the Explanation object
    shap.plots.waterfall(shap_vals_for_plot[row_idx], show=False, max_display=10)
    st.pyplot(fig2)
    plt.clf()
except Exception as e:
    st.warning(f"SHAP explanation failed for this model type: {e}")

st.subheader("4. LIME — local explanation")

try:
    lime_explainer = LimeTabularExplainer(
        X_encoded.values,
        feature_names=X_encoded.columns.tolist(),
        class_names=[str(c) for c in np.unique(y)],
        mode="classification",
    )
    lime_row = st.number_input("Row index for LIME", 0, len(X_encoded) - 1, 0, key="lime_row")
    exp = lime_explainer.explain_instance(
        X_encoded.values[lime_row],
        model.predict_proba,
        num_features=10,
    )
    st.components.v1.html(exp.as_html(), height=400, scrolling=True)
except Exception as e:
    st.warning(f"LIME explanation failed: {e}")


st.subheader("5. Bias audit")

groups = sensitive.unique().tolist()
if len(groups) < 2:
    st.info("Sensitive attribute needs at least 2 groups for a bias comparison.")
else:
    c1, c2 = st.columns(2)
    with c1:
        privileged = st.selectbox("Privileged group", groups, index=0)
    with c2:
        unprivileged = st.selectbox("Unprivileged group", [g for g in groups if g != privileged], index=0)

    y_pred_arr = pd.Series(preds).reset_index(drop=True)
    y_true_arr = y.reset_index(drop=True)
    sens_arr = sensitive.reset_index(drop=True)

    di = disparate_impact_ratio(y_pred_arr, sens_arr, privileged, unprivileged)
    spd = statistical_parity_difference(y_pred_arr, sens_arr, privileged, unprivileged)
    eod = equal_opportunity_difference(y_true_arr, y_pred_arr, sens_arr, privileged, unprivileged)
    acc_by_group = accuracy_by_group(y_true_arr, y_pred_arr, sens_arr)

    m1, m2, m3 = st.columns(3)
    m1.metric("Disparate Impact Ratio", f"{di:.3f}" if pd.notna(di) else "N/A",
               help="Ideal ≈ 1.0. Below 0.8 is a common 'four-fifths rule' red flag.")
    m2.metric("Statistical Parity Diff.", f"{spd:.3f}" if pd.notna(spd) else "N/A",
               help="Ideal ≈ 0. Positive/negative indicates favor toward one group.")
    m3.metric("Equal Opportunity Diff.", f"{eod:.3f}" if pd.notna(eod) else "N/A",
               help="Ideal ≈ 0. Compares true-positive rates across groups.")

    st.markdown("**Accuracy by group**")
    st.bar_chart(pd.Series(acc_by_group))

    flags = []
    if pd.notna(di) and (di < 0.8 or di > 1.25):
        flags.append("⚠️ Disparate Impact Ratio outside the 0.8–1.25 fair-range.")
    if pd.notna(spd) and abs(spd) > 0.1:
        flags.append("⚠️ Statistical Parity Difference exceeds ±0.1.")
    if pd.notna(eod) and abs(eod) > 0.1:
        flags.append("⚠️ Equal Opportunity Difference exceeds ±0.1.")

    if flags:
        st.error("Potential bias detected:\n\n" + "\n".join(flags))
    else:
        st.success("No major bias flags detected on the thresholds used above.")

st.divider()
st.caption(
    "EthicX MVP — SHAP/LIME explainability + rule-of-thumb fairness metrics "
    "(Disparate Impact, Statistical Parity, Equal Opportunity). "
    "Thresholds are illustrative, not legal/compliance guidance."
)