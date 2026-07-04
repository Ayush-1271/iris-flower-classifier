"""
train.py
--------
Trains a classifier on the classic Iris dataset (Fisher, 1936 / UCI Machine
Learning Repository - the same data distributed on Kaggle as "Iris Species").

Pipeline:
    1. Load data
    2. EDA (saved as PNG plots)
    3. Feature engineering (derived petal/sepal ratio features)
    4. Train/test split
    5. Hyperparameter tuning (GridSearchCV) on a RandomForestClassifier
       wrapped in a scikit-learn Pipeline (StandardScaler + model)
    6. Evaluation (accuracy, classification report, confusion matrix)
    7. Save the fitted pipeline to iris_model.pkl
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "petal_area",
    "sepal_area",
]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_data() -> pd.DataFrame:
    """Load the classic Iris dataset (150 rows, 3 balanced classes)."""
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    df["species"] = df["target"].map(dict(enumerate(TARGET_NAMES)))
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Generate a couple of EDA plots and save them as PNGs."""
    sns.set_theme(style="whitegrid")

    pairplot = sns.pairplot(
        df,
        vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        hue="species",
        diag_kind="hist",
    )
    pairplot.savefig("eda_pairplot.png", dpi=110)
    plt.close("all")

    plt.figure(figsize=(6, 5))
    corr = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].corr()
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=110)
    plt.close("all")

    print("EDA plots saved: eda_pairplot.png, eda_correlation_heatmap.png")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features that tend to help tree-based models."""
    df = df.copy()
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )


def train_and_tune(X_train, y_train) -> GridSearchCV:
    pipeline = build_pipeline()
    # Kept intentionally small: Iris only has 150 rows, so a big forest adds
    # nothing but file size. This grid is enough to tune around ~98% accuracy
    # while keeping the saved pipeline under ~100KB.
    param_grid = {
        "clf__n_estimators": [30, 60, 100],
        "clf__max_depth": [3, 5, None],
        "clf__min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=TARGET_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (accuracy={acc:.3f})")
    plt.tight_layout()
    plt.savefig("eda_confusion_matrix.png", dpi=110)
    plt.close("all")

    return {"accuracy": acc, "classification_report": report}


def main():
    print("Loading data...")
    df = load_data()

    print("Running EDA...")
    run_eda(df)

    print("Engineering features...")
    df = engineer_features(df)

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Tuning hyperparameters with GridSearchCV...")
    grid = train_and_tune(X_train, y_train)
    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV accuracy: %.4f" % grid.best_score_)

    print("Evaluating on held-out test set...")
    metrics = evaluate(best_model, X_test, y_test)
    print("Test accuracy: %.4f" % metrics["accuracy"])
    print(json.dumps(metrics["classification_report"], indent=2))

    # Refit best model on the FULL dataset before shipping it, so the
    # deployed API benefits from all 150 labeled samples.
    print("Refitting best pipeline on full dataset...")
    final_model = build_pipeline().set_params(**grid.best_params_)
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
        },
        "iris_model.pkl",
    )
    print("Saved trained pipeline to iris_model.pkl")


if __name__ == "__main__":
    main()
