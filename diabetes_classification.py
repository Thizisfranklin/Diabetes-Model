"""
diabetes_classification.py
==========================

This script is an experimental end‑to‑end machine‑learning workflow on the
Pima‑Indians diabetes dataset.  The following steps are covered:

1. **Data loading and inspection** – read the CSV file and display
   basic statistics.
2. **Data cleaning** – treat zeros in selected physiological
   variables as missing values, create missingness indicators and
   impute the missing values using medians.
3. **Feature engineering** – derive a glucose–insulin ratio and an
   age–BMI interaction term to capture non‑linear relationships.
4. **Train/test split** – create a stratified 80/20 split of the
   dataset.
5. **Model training** – fit a decision tree (with hyper‑parameter
   optimisation), random forest (with hyper‑parameter optimisation) and
   K‑nearest neighbours classifier (tuning the number of neighbours).
6. **Evaluation** – compute accuracy, precision and recall for
   each model and report the best hyper‑parameters.

Running this script will print the performance metrics and best
hyper‑parameters for each classifier.  Images such as histograms or
heatmaps can be generated separately if desired (see the report for
examples).

Usage:
    python diabetes_classification.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load the diabetes dataset from a CSV file.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file containing the diabetes data.

    Returns
    -------
    df : pd.DataFrame
        Loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean and prepare the diabetes dataset for modelling.

    The preprocessing steps include:

    1. Identify columns in which zeros represent missing values.
    2. Create binary indicator columns for missingness (1 if the
       original value was zero, 0 otherwise).
    3. Replace zeros with NaN and impute missing values using the
       median of each column.
    4. Create engineered features: Glucose/Insulin ratio and
       Age*BMI interaction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw diabetes dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix after preprocessing and feature engineering.
    y : pd.Series
        Target vector (Outcome column).
    """
    df = df.copy()
    target = df['Outcome']

    # Columns where a zero value is physiologically implausible and
    # therefore indicates a missing measurement.  Pregnancies and
    # Outcome can legitimately be zero and are not treated as missing.
    invalid_zero_cols: List[str] = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
    ]

    # Create a missingness indicator column for each variable with
    # disguised missing values.  A value of 1 means the original entry
    # was zero (missing); 0 means it contained a valid measurement.
    for col in invalid_zero_cols:
        df[f'{col}_Missing'] = (df[col] == 0).astype(int)

    # Replace zeros with NaN to mark them as missing.  This allows
    # pandas/scikit‑learn to handle them properly during imputation.
    df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

    # Compute medians for each invalid_zero column and fill NaNs with
    # these values.  Median imputation is robust to outliers and
    # preserves the central tendency of the data.
    median_values = df[invalid_zero_cols].median()
    df[invalid_zero_cols] = df[invalid_zero_cols].fillna(median_values)

    # Feature engineering: Glucose/Insulin ratio and Age*BMI interaction.
    # A small constant is added to the denominator in the ratio to
    # avoid division by zero (although insulin should no longer be
    # zero after imputation).
    df['GlucoseInsulinRatio'] = df['Glucose'] / (df['Insulin'] + 1e-6)
    df['AgeBMIInteraction'] = df['Age'] * df['BMI']

    # Separate features and target.  Drop the original Outcome column.
    X = df.drop(columns=['Outcome'])
    y = target.copy()
    return X, y


def train_decision_tree(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Train a Decision Tree classifier with grid‑search hyper‑parameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Testing features.
    y_test : pd.Series
        Testing labels.

    Returns
    -------
    metrics : dict
        Accuracy, precision and recall of the best decision tree.
    best_params : dict
        Hyper‑parameters of the best decision tree found by grid search.
    """
    # Define the parameter grid.  We search over splitting criteria
    # (Gini vs. entropy), tree depth and minimum samples per split.
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 9],
        'min_samples_split': [2, 5, 10],
    }
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    best_dt: DecisionTreeClassifier = grid_search.best_estimator_

    # Evaluate on the test set
    y_pred = best_dt.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }
    return metrics, grid_search.best_params_


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, float], Dict[str, object], pd.Series]:
    """Train a Random Forest classifier with grid‑search hyper‑parameter tuning.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data

    Returns
    -------
    metrics : dict
        Accuracy, precision and recall of the best random forest.
    best_params : dict
        Hyper‑parameters of the best model.
    feature_importances : pd.Series
        Importance of each feature, sorted in descending order.
    """
    param_grid = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf: RandomForestClassifier = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }
    # Capture feature importances
    importances = pd.Series(
        best_rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    return metrics, grid_search.best_params_, importances


def train_knn(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
    y_test: pd.Series, k_range: Tuple[int, int] = (1, 20)
) -> Tuple[Dict[str, float], int, List[float]]:
    """Train a K‑Nearest Neighbours classifier and select the optimal k.

    The data are scaled using StandardScaler because KNN relies on
    Euclidean distances and is sensitive to the scale of the input
    variables.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    k_range : tuple
        Inclusive range of k values to evaluate (min_k, max_k).

    Returns
    -------
    metrics : dict
        Accuracy, precision and recall for the best k.
    best_k : int
        Number of neighbours producing the highest accuracy.
    k_accuracies : list
        Accuracy values for each evaluated k.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start_k, end_k = k_range
    ks = list(range(start_k, end_k + 1))
    accuracies: List[float] = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    best_idx = int(np.argmax(accuracies))
    best_k = ks[best_idx]
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_pred_best = best_knn.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_best),
        'precision': precision_score(y_test, y_pred_best),
        'recall': recall_score(y_test, y_pred_best),
    }
    return metrics, best_k, accuracies


def main() -> None:
    """Main execution function.

    Loads the data, preprocesses it, splits into train/test sets,
    trains three types of classifiers and prints their performance
    metrics and best hyper‑parameters.  Results are also stored in a
    JSON file for convenience.
    """
    dataset_path = pathlib.Path('diabetes.csv')
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file '{dataset_path}' not found.  Place the CSV in the same directory as this script."
        )

    df = load_data(dataset_path)
    X, y = preprocess_data(df)

    # Train/test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision Tree
    dt_metrics, dt_params = train_decision_tree(
        X_train, y_train, X_test, y_test
    )

    # Random Forest
    rf_metrics, rf_params, rf_importances = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # KNN
    knn_metrics, best_k, knn_accs = train_knn(
        X_train, y_train, X_test, y_test, k_range=(1, 20)
    )

    # Summarise results
    results = {
        'Decision Tree': {
            'accuracy': dt_metrics['accuracy'],
            'precision': dt_metrics['precision'],
            'recall': dt_metrics['recall'],
            'best_params': dt_params,
        },
        'Random Forest': {
            'accuracy': rf_metrics['accuracy'],
            'precision': rf_metrics['precision'],
            'recall': rf_metrics['recall'],
            'best_params': rf_params,
            'top_features': rf_importances.head(10).to_dict(),
        },
        'KNN': {
            'accuracy': knn_metrics['accuracy'],
            'precision': knn_metrics['precision'],
            'recall': knn_metrics['recall'],
            'best_k': best_k,
            'k_accuracies': {
                str(k): acc for k, acc in zip(range(1, 21), knn_accs)
            },
        },
    }

    # Write results to a JSON file for easy inspection
    results_path = pathlib.Path('model_performance.json')
    with results_path.open('w') as f:
        json.dump(results, f, indent=4)

    # Print results to console
    print("Performance summary:\n")
    for model_name, info in results.items():
        print(f"{model_name}:\n"
              f"  Accuracy:  {info['accuracy']:.3f}\n"
              f"  Precision: {info['precision']:.3f}\n"
              f"  Recall:    {info['recall']:.3f}\n"
              f"  Hyper‑parameters: {info.get('best_params', {'k': info.get('best_k')})}\n")

    # Optionally print top feature importances
    top_features = results['Random Forest']['top_features']
    print("Top features (Random Forest):")
    for feat, imp in top_features.items():
        print(f"  {feat:<20} {imp:.3f}")


if __name__ == '__main__':
    main()
