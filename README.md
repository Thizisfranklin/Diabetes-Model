# Diabetes Risk Prediction (Pima Indians)

I built a small, end-to-end project to predict diabetes risk from routine clinical measurements. The goal is simple: **catch high-risk patients early** so a care team can prioritize follow-ups and preventive care.

---

## Why this exists

- Diabetes is expensive to treat late. A lightweight risk score can help triage patients for labs, counseling, and monitoring.
- I wanted a compact project that shows my DS workflow: EDA → cleaning → feature engineering → modeling → evaluation → takeaways.

---

## Data

- **Source:** Pima Indians Diabetes dataset (768 adult female patients ≥21).
- **Target:** `Outcome` (1 = diabetic, 0 = non-diabetic).
- **Notes:** Several physiological fields have zeros (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`). Adults don’t have zero values here, so I treat zeros as **missing**.

---

## What I did (short version)

1. **EDA:** Distributions, outliers, simple correlations. Class split ≈ 35% positive / 65% negative. Glucose and BMI correlate most with `Outcome`.
2. **Cleaning:** Replace impossible zeros with `NaN`, **median impute**, and add `*_Missing` flags so models can learn from the pattern of missingness.
3. **Features:**  
   - `GlucoseInsulinRatio = Glucose / (Insulin + 1e-6)` (insulin resistance proxy)  
   - `AgeBMIInteraction = Age * BMI` (compounded risk)
4. **Modeling (multiple candidates):**  
   I **built, fit, and compared several models** to select the strongest baseline for this task:
   - **Decision Tree** (tuned: `criterion`, `max_depth`, `min_samples_split`)
   - **Random Forest** (tuned: `n_estimators`, `max_depth`, `min_samples_split`)
   - **K-Nearest Neighbors (KNN)** (tuned: `k = 1..20`, with **standardization**)
5. **Evaluation:** Accuracy, Precision, **Recall** (I prioritize **recall** to miss fewer true positives), using a stratified 80/20 split.

---

## Results (held-out test)

| Model         | Accuracy | Precision | Recall |
|---------------|---------:|----------:|-------:|
| Decision Tree |   0.695  |    0.667  |  0.259 |
| Random Forest |   0.740  |    0.675  |  0.500 |
| KNN (k=16)    |   0.753  |    0.700  |  0.519 |

**Top RF features:** Glucose, AgeBMIInteraction, BMI, Insulin, Age.

**Choice:** I’d ship the **Random Forest** as the baseline. It trades a bit of accuracy for better recall and exposes feature importances for basic explainability. If recall must go higher, I’d tune class weights or thresholds and measure the downstream cost of false positives.

---

## How to run

### Colab
1. Upload `diabetes.csv` to `/content`.
2. Run `Lab7_Osualaaham.ipynb` or paste the single-cell version from the repo.
3. Check the printed metrics.

### Local
```bash
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -U pandas numpy scikit-learn matplotlib seaborn
# put diabetes.csv in the repo root
python Lab7_Osualaaham.ipynb
