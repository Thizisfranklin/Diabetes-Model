# DiabetesWatch: diabetes risk predictor

# Diabetes Risk Prediction (Pima Indians)

I built an end-to-end project to predict diabetes risk from routine clinical measurements. The goal is simple: **catch high-risk patients early** so care teams can prioritize follow-ups and preventive care.

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

## What I did (short version)

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

**Plain-English takeaways**
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
2. Run `diabetes_classification.py` (or paste the single-cell version).
3. Review printed metrics and plots (ROC/AUC included).

### Local
```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -U pandas numpy scikit-learn matplotlib seaborn
# put diabetes.csv in the repo root
python diabetes_classification.py
