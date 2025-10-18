# DiabetesWatch

I built an end-to-end project to predict diabetes risk from routine clinical measurements. The goal is simple: **catch high-risk patients early** so care teams can prioritize follow-ups and preventive care.

## Brief Description
This project tackles early identification of diabetes risk using routine clinical measurements (Pima Indians dataset). I cleaned physiologically “impossible” zeros (e.g., Glucose, BMI) by imputing them, engineered risk-focused features (Glucose/Insulin ratio, Age×BMI), and built multiple models—Decision Tree, Random Forest, and K-Nearest Neighbors—to find a strong baseline. Models were evaluated on Accuracy, Precision, Recall, and AUC, with ROC curves used to compare discrimination. The goal is practical: **flag high-risk patients early** so care teams can prioritize follow-ups and preventive care.

## What I Built
- End-to-end DS workflow: EDA → cleaning/imputation → feature engineering → model training/tuning → evaluation/visualization  
- Visuals: distributions, boxplots, correlation heatmap, ROC curves  
- Reproducible script/notebook for training and metrics export

## Tools Used
- Python (3.10+), pandas, numpy, scikit-learn  
- Matplotlib, seaborn (plots)  
- Jupyter/Colab for experiments; Git/GitHub for versioning

---

## Why this exists

- Treating diabetes late is expensive; a lightweight risk score helps triage patients for labs, counseling, and monitoring.
- This project showcases my DS workflow: **EDA → cleaning → feature engineering → modeling → evaluation → takeaways** in a format anyone can read.

---

## Data

- **Source:** Pima Indians Diabetes dataset (768 adult female patients ≥21).
- **Target:** `Outcome` (1 = diabetic, 0 = non-diabetic).
- **Important note:** Several physiological fields (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) contain zeros that are not biologically plausible for living adults. I treat these zeros as **missing**.

---

## What I did (short overview)

1. **EDA:** Looked at distributions, outliers, correlations. Class split ≈ **35% positive / 65% negative**. `Glucose` and `BMI` correlated most with `Outcome`.
2. **Cleaning:** Replaced impossible zeros with `NaN`, **median-imputed**, and added `*_Missing` flags so models can learn from missingness patterns.
3. **Features:**
   - `GlucoseInsulinRatio = Glucose / (Insulin + 1e-6)` (insulin-resistance proxy)
   - `AgeBMIInteraction = Age * BMI` (compounded risk)
4. **Modeling (multiple candidates):** I **built, fit, and compared several models** to select the strongest baseline:
   - **Decision Tree** (tuned `criterion`, `max_depth`, `min_samples_split`)
   - **Random Forest** (tuned `n_estimators`, `max_depth`, `min_samples_split`)
   - **K-Nearest Neighbors (KNN)** (tuned `k = 1..20`, with **standardization**)
5. **Evaluation:** Accuracy, Precision, **Recall**, and **AUC** on a **stratified 80/20 split**.

---

## Visuals included

- Histograms for **Glucose** and **BloodPressure** distributions  
- Boxplot of **Glucose vs Outcome** (Outcome=1 shows higher glucose)  
- **ROC curves** for all three models  
- Bar plots for **Accuracy/Precision/Recall** and a separate **AUC** comparison

---

## Results (held-out test)

| Model         | Accuracy | Precision | Recall | AUC   |
|---------------|---------:|----------:|-------:|------:|
| Decision Tree |   0.695  |    0.667  |  0.259 | 0.751 |
| Random Forest |   0.734  |    0.651  |  0.519 | 0.805 |
| KNN (k=8)     |   0.747  |    0.692  |  0.500 | 0.780 |

**Takeaways**
- **KNN (k=8)**: best **Accuracy** and **Precision**, solid **Recall**  
- **Random Forest**: best **Recall** and **AUC** (strongest at finding positives and separating classes)  
- **Decision Tree**: weakest overall but useful as a transparent baseline

**Top RF features:** Glucose, AgeBMIInteraction, BMI (with Insulin and Age also contributing)

---

## “Best” model depends on the goal

- **Minimize false negatives (miss fewer true diabetics):** **Random Forest** (higher **Recall/AUC**)  
- **Minimize false positives / maximize overall correctness:** **KNN (k=8)** (higher **Accuracy/Precision**)  
- **Maximum transparency:** **Decision Tree** baseline

My default **ship-ready baseline** is **Random Forest** for its recall/AUC profile and feature importances. If precision needs to be higher, I’d deploy KNN with a tuned threshold aligned to downstream costs.

---

## How to run

### Colab
1. Upload `diabetes.csv` to `/content`.
2. Run `Lab7_Osualaaham.ipynb` (or paste the single-cell version).
3. Review printed metrics and plots (ROC/AUC included).

### Local

#### Run the script
```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -U pandas numpy scikit-learn matplotlib seaborn
# put diabetes.csv in the repo root
python diabetes_classification.py
```

#### (optional) install Jupyter
```
pip install -U jupyter
```

#### launch and open the notebook
```
jupyter lab             # or: jupyter notebook
# then open: Lab7_Osualaaham.ipynb
```

---

## ⚠️ Potential Biases, Limitations & How to Improve

**Population & sampling bias**
- *Issue:* The dataset contains **adult female Pima-Indian patients** only; results may not generalize to other demographics or care settings.
- *Improve:* Validate on external, multi-site cohorts with diverse age/sex/ethnicity; report subgroup metrics.

**Measurement bias / disguised missingness**
- *Issue:* Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` are **not physiologically plausible** and act as hidden missing data.
- *Improve:* Use **multiple imputation** (e.g., MICE) with sensitivity analyses; push for better upstream data capture; log data quality KPIs.


---

## 📚 References
- **scikit-learn documentation**
  - Preprocessing and scaling: https://scikit-learn.org/stable/modules/preprocessing.html  
  - Nearest neighbors overview: https://scikit-learn.org/stable/modules/neighbors.html  
  - ROC/AUC & model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Dataset sources**
  - Pima Indians Diabetes (UCI ML Repository): https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes  
  - Pima Indians Diabetes (Kaggle mirror): https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **EDA note on disguised missingness (zeros → NaN)**
  - Pima Indians Diabetes — Analysis & Predictions (obrunet.github.io): https://obrunet.github.io/pima-indians-diabetes-analysis-predictions
