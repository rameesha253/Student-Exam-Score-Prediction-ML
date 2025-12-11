import gradio as gr
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor

# ---------- Helper Functions ----------
def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1)), X])

def fit_reg(X, y):
    XtX = X.T @ X
    XtY = X.T @ y
    ridge = 1e-8 * np.eye(XtX.shape[0])  # stability
    beta = np.linalg.inv(XtX + ridge) @ XtY
    return beta

def predict(X, beta):
    return X @ beta

def MAE(y, yh):
    return float(np.mean(np.abs(y - yh)))

def RMSE(y, yh):
    return float(np.sqrt(np.mean((y - yh)**2)))

def R2(y, yh):
    ss_res = np.sum((y - yh)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res/ss_tot) if ss_tot != 0 else 0.0

# ---------- Prediction Function ----------
def run_dashboard(file, rq_choice):
    if file is None:
        return "Please upload marks_dataset.xlsx", None

    # Load entire workbook
    xls = pd.ExcelFile(file.name)
    dfs = [pd.read_excel(file.name, sheet_name=s) for s in xls.sheet_names]
    data = pd.concat(dfs, ignore_index=True)

    # Clean column names
    data.columns = [c.strip() for c in data.columns]

    # Convert all numeric cols safely
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="ignore")

    # ---------- RQ-based allowed predictors ----------
    if rq_choice == "RQ1 - Predict S-I":
        target = "S-I"
        feats = [c for c in data.columns if c.startswith("As:") or c.startswith("Qz:")]

    elif rq_choice == "RQ2 - Predict S-II":
        target = "S-II"
        feats = [c for c in data.columns if c.startswith("As:") or c.startswith("Qz:")]
        if "S-I" in data.columns:
            feats.append("S-I")

    elif rq_choice == "RQ3 - Predict Final":
        target = "Final"
        feats = [c for c in data.columns if c.startswith("As:") or c.startswith("Qz:")]
        if "S-I" in data.columns:
            feats.append("S-I")
        if "S-II" in data.columns:
            feats.append("S-II")

    # Keep only available columns
    feats = [f for f in feats if f in data.columns]

    # Build working dataset
    df = data[feats + [target]].copy()

    # Convert everything to numeric and fill missing with 0
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Drop completely-zero rows
    mask_zero = (df.sum(axis=1) == 0)
    df = df[~mask_zero]

    # ---------- Manual Shuffle + Split ----------
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    test_size = int(len(df) * 0.2)

    test = df.iloc[:test_size].reset_index(drop=True)
    train = df.iloc[test_size:].reset_index(drop=True)

    X_train = train[feats].values
    y_train = train[target].values
    X_test = test[feats].values
    y_test = test[target].values

    # ---------- Best Single Predictor ----------
    cors = {}
    for c in feats:
        if np.std(train[c]) > 0:
            cors[c] = abs(np.corrcoef(train[c], y_train)[0,1])
        else:
            cors[c] = 0

    best_feat = max(cors, key=cors.get)

    # ---------- Simple Model ----------
    X1_train = add_bias(train[[best_feat]].values)
    X1_test = add_bias(test[[best_feat]].values)

    beta1 = fit_reg(X1_train, y_train)
    y1_train = predict(X1_train, beta1)
    y1_test = predict(X1_test, beta1)

    # ---------- Multiple Model ----------
    X2_train = add_bias(X_train)
    X2_test = add_bias(X_test)

    beta2 = fit_reg(X2_train, y_train)
    y2_train = predict(X2_train, beta2)
    y2_test = predict(X2_test, beta2)

    # ---------- Dummy Model ----------
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    yd_train = dummy.predict(X_train)
    yd_test = dummy.predict(X_test)

    # ---------- Comparison Table ----------
    results = pd.DataFrame([
        ["Simple Model", best_feat,
         MAE(y_train, y1_train), MAE(y_test, y1_test),
         RMSE(y_train, y1_train), RMSE(y_test, y1_test),
         R2(y_train, y1_train), R2(y_test, y1_test)],

        ["Multiple Model", "All Allowed Features",
         MAE(y_train, y2_train), MAE(y_test, y2_test),
         RMSE(y_train, y2_train), RMSE(y_test, y2_test),
         R2(y_train, y2_train), R2(y_test, y2_test)],

        ["Dummy Model", "--",
         MAE(y_train, yd_train), MAE(y_test, yd_test),
         RMSE(y_train, yd_train), RMSE(y_test, yd_test),
         R2(y_train, yd_train), R2(y_test, yd_test)]
    ], columns=["Model","Features","Train MAE","Test MAE","Train RMSE","Test RMSE","Train R2","Test R2"])

    return results, "Prediction updated successfully!"

# ---------- Gradio UI ----------
input_file = gr.File(label="Upload marks_dataset.xlsx")

rq_dropdown = gr.Dropdown(
    ["RQ1 - Predict S-I", "RQ2 - Predict S-II", "RQ3 - Predict Final"],
    label="Choose Research Question"
)

demo = gr.Interface(
    fn=run_dashboard,
    inputs=[input_file, rq_dropdown],
    outputs=[gr.Dataframe(label="Model Comparison Table"),
             gr.Textbox(label="Status")],
    title="Student Marks Prediction — Gradio Dashboard",
    description="Beginner-friendly dashboard to compare Simple, Multiple, and Dummy regression models."
)

demo.launch()
