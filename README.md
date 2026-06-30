# 🫀 Heart Disease Prediction — Comparative ML Study

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-teal?style=flat-square)

> End-to-end Data Science project — from raw data with quality issues to honest, validated ML results. Covers EDA, duplicate detection, preprocessing, and comparison of three models with cross-validation.

---

## 📂 Project Structure
## Project Structure

```text
heart-disease-prediction/
├── data/
│   └── heart.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
└── requirements.txt
```

* **data/** – Contains the heart disease dataset used for analysis and model training.
* **notebooks/** – Jupyter Notebook containing data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and visualization.
* **requirements.txt** – Lists all Python dependencies required to run the project.

```
```


---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | UCI ML Repository / Kaggle |
| Raw samples | 1,025 patients |
| After cleaning | 302 patients |
| Duplicates removed | 723 rows |
| Features | 13 clinical features |
| Target | Binary — 1 (Disease) / 0 (No Disease) |
| Class balance | Well balanced — no resampling needed |

---

## 🔍 EDA — Key Findings

- ✅ No missing values across any feature
- 📈 **Thalach** (max heart rate) showed the strongest correlation with the target
- 🧓 Younger patients slightly more represented in the disease class
- 🔬 Cholesterol and resting BP had weaker individual correlations but contributed meaningfully in combination

---

## ⚙️ Preprocessing

- 🧹 **723 duplicate rows detected and removed** — leaving them caused 100% (misleading) test scores
- 🔢 Categorical features one-hot encoded: `cp` `restecg` `slope` `ca` `thal`
- 📏 Numerical features scaled with `StandardScaler`
- ✂️ 80/20 stratified train/test split (`random_state=42`)

---

## 🤖 Models

| Model | Why Included |
|---|---|
| Logistic Regression | Linear baseline — fast and interpretable |
| Random Forest | Ensemble — handles non-linearity, gives feature importance |
| XGBoost | Gradient boosting — typically strongest on tabular data |

---

## 📈 Results

### 5-Fold Cross-Validation (AUC)

| Model | Mean AUC | Std |
|---|---|---|
| ⭐ Logistic Regression | **0.9243** | 0.0358 |
| XGBoost | 0.8922 | 0.0403 |
| Random Forest | 0.8844 | 0.0523 |

### Test Set Performance

| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| ⭐ Logistic Regression | **85.25%** | **86.15%** | **0.9026** |
| Random Forest | 78.69% | 80.00% | 0.8907 |
| XGBoost | 78.69% | 79.37% | 0.8528 |

---

## 💡 Key Findings

**🏆 Best model: Logistic Regression**
Outperformed both ensemble methods across every metric. Suggests the relationship between clinical features and heart disease is largely linear in this dataset.

**🌳 Why tree models underperformed**
Random Forest and XGBoost typically dominate tabular data — but with only 302 clean rows and a linear signal, simpler models generalised better. Model complexity is not always the answer.

**🔑 Top features (Random Forest)**
`thalach` · `oldpeak` · `chol` — consistent with EDA and medically plausible.

**🧹 Data quality catch**
Initial runs showed suspiciously perfect scores (100% accuracy, 1.0 AUC). Investigation revealed 723 duplicate rows causing test set contamination. Removing them revealed true model performance — a real-world DS lesson more valuable than any metric.

---

## 🏁 Conclusion

Clinical features alone can predict heart disease with strong accuracy (~90% AUC). The most important lesson from this project was identifying data quality issues that produced misleading perfect scores. Future work could explore SHAP values for clinical explainability and hyperparameter tuning on a larger independent dataset.

---

## 🛠️ Tools & Libraries

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn` · `jupyter`

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

---

<p align="center"><i>Built by Abdullah Waheed · Data Science & ML Portfolio</i></p>
