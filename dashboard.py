import gradio as gr
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor

# ---------- Helper functions (same as notebook) ----------
def add_bias(X): 
    return np.hstack([np.ones((X.shape[0],1)), X])

def fit_reg(X, y):
    XtX = X.T @ X
    XtY = X.T @ y
    beta = np.linalg.inv(XtX) @ XtY
    return beta

def predict(X, beta): 
    return X @ beta

def MAE(y, yhat): 
    return float(np.mean(np.abs(y - yhat)))

def RMSE(y, yhat):
    return float(np.sqrt(np.mean((y - yhat)**2)))

def R2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return float(1 - ss_res/ss_tot) if ss_tot != 0 else 0.0

# ---------- Main prediction function ----------
def run_dashboard(file, rq_choice):

    if file is None:
        return "Please upload marks_dataset.xlsx", None

    # Load and merge all sheets
    xls = pd.ExcelFile(file.name)
    dfs = [pd.read_excel(file.name, sheet_name=s) for s in xls.sheet_names]
    data = pd.concat(dfs, ignore_index=True)

    # Clean column names
    data.columns = [c.strip() for c in data.columns]

    # Select features based on RQ
    rq_map = {
        "RQ1 - Predict S-I": (
            "S-I",
            ["As:1","As:2","As:3","As:4",
             "Qz:1","Qz:2","Qz:3","Qz:4","Qz:5","Qz:6",
             "S-I:1","S-I:2","S-I:3"]
        ),
        "RQ2 - Predict S-II": (
            "S-II",
            ["As:1","As:2","As:3","As:4",
             "Qz:1","Qz:2","Qz:3","Qz:4","Qz:5","Qz:6",
             "S-I","S-II:1","S-II:2","S-II:3"]
        ),
        "RQ3 - Predict Final": (
            "Final",
            ["As:1","As:2","As:3","As:4",
             "Qz:1","Qz:2","Qz:3","Qz:4","Qz:5","Qz:6",
             "S-I","S-II",
             "Final:1","Final:2","Final:3","Final:4","Final:5"]
        )
    }

    target, feats = rq_map[rq_choice]
    feats = [f for f in feats if f in data.columns]

    df = data[feats + [target]].copy()
    df = df.fillna(df.median())

    # Manual shuffle + split
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    test_size = int(len(df) * 0.2)
    test = df[:test_size]
    train = df[test_size:]

    X_train = train[feats].values
    y_train = train[target].values
    X_test = test[feats].values
    y_test = test[target].values

    # Best single predictor
    cors = {}
    for c in feats:
        if np.std(train[c]) > 0:
            cors[c] = abs(np.corrcoef(train[c], y_train)[0,1])
        else:
            cors[c] = 0
    best_feat = max(cors, key=cors.get)

    # Simple Model
    X1_train = add_bias(train[[best_feat]].values)
    X1_test = add_bias(test[[best_feat]].values)
    beta1 = fit_reg(X1_train, y_train)
    y1_train = predict(X1_train, beta1)
    y1_test = predict(X1_test, beta1)

    # Multiple Model
    X2_train = add_bias(X_train)
    X2_test = add_bias(X_test)
    beta2 = fit_reg(X2_train, y_train)
    y2_train = predict(X2_train, beta2)
    y2_test = predict(X2_test, beta2)

    # Dummy Model
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    yd_train = dummy.predict(X_train)
    yd_test = dummy.predict(X_test)

    # Prepare comparison table
    results = pd.DataFrame([
        ["Simple Model", best_feat, MAE(y_train,y1_train), MAE(y_test,y1_test), RMSE(y_train,y1_train), RMSE(y_test,y1_test), R2(y_train,y1_train), R2(y_test,y1_test)],
        ["Multiple Model", "All Features", MAE(y_train,y2_train), MAE(y_test,y2_test), RMSE(y_train,y2_train), RMSE(y_test,y2_test), R2(y_train,y2_train), R2(y_test,y2_test)],
        ["Dummy Model", "--", MAE(y_train,yd_train), MAE(y_test,yd_test), RMSE(y_train,yd_train), RMSE(y_test,yd_test), R2(y_train,yd_train), R2(y_test,yd_test)]
    ], columns=["Model","Features","Train MAE","Test MAE","Train RMSE","Test RMSE","Train R2","Test R2"])

    return results, "Prediction completed successfully!"

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
