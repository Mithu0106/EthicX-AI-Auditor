# EthicX MVP

A 1-file Streamlit app that audits a classifier for **explainability** (SHAP + LIME)
and **bias** (Disparate Impact, Statistical Parity, Equal Opportunity) — built to be
demo-able in under 2 hours.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Try it instantly with the included sample

Two files are bundled so you can demo without training anything:

- `loan_sample_dataset.csv` — synthetic loan-approval data with a **deliberately
  injected mild bias** favoring the "Male" group (for demo purposes only).
- `loan_sample_model.pkl` — a RandomForestClassifier already trained on that dataset.

In the app sidebar:
1. Upload `loan_sample_dataset.csv`
2. Upload `loan_sample_model.pkl`
3. Target column → `loan_approved`
4. Sensitive attribute → `gender`

You should see the bias flags trigger on Disparate Impact Ratio.

## What it does

1. **Dataset preview** — quick sanity check of the uploaded CSV.
2. **Model performance** — accuracy + confusion matrix.
3. **SHAP** — global feature-importance bar chart + a per-row waterfall plot.
4. **LIME** — local explanation for any single row, rendered inline.
5. **Bias audit** — pick a privileged/unprivileged group on any sensitive column:
   - Disparate Impact Ratio (flag if outside 0.8–1.25, the "four-fifths rule")
   - Statistical Parity Difference (flag if |diff| > 0.1)
   - Equal Opportunity Difference (flag if |diff| > 0.1)
   - Accuracy by group, as a bar chart

## Bring your own model

- Upload any `.pkl` / `.joblib` binary classifier that exposes `.predict()` and
  ideally `.predict_proba()`.
- **Important:** the model must have been trained on features that match this
  app's encoding — categorical columns are one-hot encoded with
  `pd.get_dummies(..., drop_first=False).astype(float)`. If your model was
  trained differently, predictions will fail loudly with a clear error rather
  than silently producing garbage.
- No model handy? Check **"Train a quick RandomForest demo model"** in the
  sidebar and the app will train one on 75% of your uploaded data on the fly.


## Known MVP limitations :

- One-hot encoding is naive — no handling of high-cardinality categoricals,
  ordinal encoding, or missing values beyond pandas defaults.
- SHAP uses `shap.Explainer`'s auto-selected algorithm; for non-tree models
  this can be slow on large datasets (consider sampling rows before SHAP for
  datasets > ~2000 rows).
- Fairness thresholds (0.8–1.25 for DI, ±0.1 for parity diffs) are illustrative
  rule-of-thumb values, not legal/regulatory compliance guidance.
- Binary classification only, single sensitive attribute at a time (no
  intersectional fairness yet — a natural v2 feature).
- No persistence/auth — every session is stateless, nothing is saved server-side.