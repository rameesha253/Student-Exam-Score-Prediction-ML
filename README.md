# 🎓 **Student Performance Prediction Using Regression Models + Gradio Dashboard**

A complete end-to-end ML project predicting **Midterm I**, **Midterm II**, and **Final Exam** scores using **regression models**, **bootstrapping**, and an interactive **Gradio dashboard**.

---

## 📌 **Project Overview**

This project analyzes anonymized student assessment data collected across **six Excel sheets**, each containing various quizzes, assignments, and exam components.
After preprocessing and merging the datasets, I built multiple regression models to answer three core research questions:

### **Research Questions**

1. **RQ1:** How accurately can we predict student marks in **Midterm I**?
2. **RQ2:** How accurately can we predict student marks in **Midterm II**?
3. **RQ3:** How accurately can we predict student marks in the **Final Examination**?

---

## 🎯 **Objectives**

* Perform **Exploratory Data Analysis (EDA)**
* Combine all six sheets into a unified dataset
* Apply domain-aware **feature selection**
* Train multiple regression models:

  * Simple Regression
  * Multiple Regression
  * Polynomial Regression (optional, depending on notebook)
* Perform **bootstrapping (500 samples)** for error confidence intervals
* Compare all models using **MAE, RMSE, and R²**
* Evaluate results against a **Dummy Regressor baseline**
* Build a **Gradio dashboard** to interactively test models

---

## 🛠 **Tools & Techniques**

* **Python**
* **Pandas, NumPy** – Data Cleaning & Preprocessing
* **Matplotlib, Seaborn** – Data Visualization
* **Scikit-learn** – DummyRegressor, Metrics
* **Manual Regression Implementation** (normal equation)
* **Bootstrapping** – 500 samples
* **Gradio** – Interactive Dashboard

---

## 🧠 **Work Flow**

### **1. Data Loading & Merging**

* Loaded all 6 Excel sheets
* Cleaned missing values (median imputation)
* Combined into one dataset

### **2. Exploratory Data Analysis**

* Distribution analysis
* Correlation heatmaps
* Outlier detection
* Feature importance exploration

### **3. Model Training**

For each research question:

* Selected domain-correct predictor variables
* Trained at least **two regression models**
* Evaluated baseline Dummy Regressor

### **4. Model Evaluation**

* Computed **MAE**, **RMSE**, **R²**
* Bootstrapped **500 samples** to estimate **95% CI for MAE**
* Compared all models in a table
* Checked overfitting/underfitting using train vs test scores

### **5. Gradio Dashboard**

The dashboard:

* Accepts Excel input
* Automatically merges sheets
* Predicts based on RQ selection
* Displays comparison table of **Simple, Multiple, Dummy** models

---

# ✅ **📊 Results Summary**

After training Simple Regression, Multiple Regression, and Dummy (baseline) models for all three research questions (RQ1, RQ2, RQ3), the following insights were drawn:

### **🔹 RQ1 — Predicting Midterm I (S-I)**

* **Multiple Regression** achieved the best performance, with significantly lower MAE & RMSE compared to Simple Regression.
* **Simple Regression** using the strongest single feature showed moderate predictive power.
* **Dummy Regressor** performed poorly, confirming that our trained models learned meaningful patterns.
* **Bootstrapping (500 samples)** showed narrow confidence intervals, indicating stable and reliable error estimates.

### **🔹 RQ2 — Predicting Midterm II (S-II)**

* **Multiple Regression** again outperformed all other models.
* The prediction accuracy improved compared to RQ1 because the presence of S-I as a predictor provided better signal.
* **Dummy model** performed at baseline level, validating the strength of our trained models.
* Bootstrapped MAE confidence intervals were consistent and showed low variance.

### **🔹 RQ3 — Predicting Final Exam Marks**

* This was the *most complex* prediction task due to multiple final exam components.
* **Multiple Regression** delivered the strongest results with high R² and lower errors.
* **Simple Regression** showed limited predictive capability due to the exam’s multi-part structure.
* Bootstrapping confirmed stable performance with reliable MAE confidence intervals.
* Overall, RQ3 demonstrated the highest benefit of using multivariate regression due to feature richness.

### **📌 Summary of Observations**

* **Multiple Regression = consistently best performer across all RQs**
* **Simple Regression = useful but limited**
* **Dummy Regressor = lowest (baseline) performance**
* **Bootstrapping = demonstrated stability and low variance in model errors**
* **No major overfitting detected** based on train–test score comparison of the best models

These results validate the effectiveness of multivariate linear relationships in academic score prediction and the importance of proper feature selection guided by domain knowledge.

---

# ✅ **🖥 How to Run the Gradio Dashboard**

### **1. Install Dependencies**

Make sure you have all required packages installed:

```bash
pip install -r requirements.txt
```

### **2. Run the Dashboard**

Open **Command Prompt (CMD)** in the project directory and run:

```bash
python dashboard.py
```

### **3. Use the Dashboard**

* Upload the **marks_dataset.xlsx** file
* Select any of the three research questions
* View the automatically generated **model comparison table**
* Check MAE, RMSE, R² for Simple, Multiple, and Dummy models

---

### **📌 Final Note**

This project demonstrates how classical regression techniques, combined with domain-aware feature engineering and statistical bootstrapping, can effectively predict student performance. The interactive Gradio dashboard makes the entire workflow accessible, interpretable, and easy to test with new datasets.

---
