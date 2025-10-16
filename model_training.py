# ...existing code...
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import joblib
import json
import sys
import sklearn
from datetime import datetime

from eda import EDAProcessor

def load_df(csv_path="loan_data.csv"):
    df = pd.read_csv(csv_path)
    return df

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")
    return preproc

def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    }

def compute_metrics(y_true, y_pred, y_proba, average_metric="macro"):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average_metric, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average_metric, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average_metric, zero_division=0)
    try:
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # binary
                p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, p)
            else:
                # multiclass: macro ovR
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        else:
            metrics["roc_auc"] = np.nan
    except Exception:
        metrics["roc_auc"] = np.nan
    return metrics

def train_and_select(csv_path="loan_data.csv", target_col="loan_status", test_size=0.2, random_state=42, out_dir="models", cv_scoring="f1_macro"):
    df = load_df(csv_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in df.columns")

    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    preproc = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    classifiers = get_classifiers()

    os.makedirs(out_dir, exist_ok=True)
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, clf in classifiers.items():
        pipe = Pipeline([("pre", preproc), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = None
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                y_proba = None
        metrics = compute_metrics(y_test, y_pred, y_proba, average_metric="macro")
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1)
            metrics[f"cv_{cv_scoring}_mean"] = float(np.mean(cv_scores))
            metrics[f"cv_{cv_scoring}_std"] = float(np.std(cv_scores))
        except Exception:
            metrics[f"cv_{cv_scoring}_mean"] = np.nan
            metrics[f"cv_{cv_scoring}_std"] = np.nan

        metrics["model"] = name
        print(f"  {name}: acc={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, roc_auc={metrics.get('roc_auc'):.3f}")
        results.append(metrics)
        # save model artifact (use .joblib) and metadata
        model_path = Path(out_dir) / f"{name}.joblib"
        joblib.dump({"pipeline": pipe, "features": X.columns.tolist()}, model_path, compress=3)
        # write metadata for reproducibility
        metadata = {
            "model": name,
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "features": X.columns.tolist(),
        }
        (Path(out_dir) / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2))

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(Path(out_dir)/"classification_results.csv", index=False)

    if results_df[f"cv_{cv_scoring}_mean"].notna().any():
        best = results_df.iloc[0]
    else:
        best = results_df.sort_values(by="f1", ascending=False).iloc[0]

    best_name = best["model"]
    best_model = joblib.load(Path(out_dir)/f"{best_name}.joblib")["pipeline"]
    best_path = Path(out_dir) / "best_model.joblib"
    joblib.dump({"pipeline": best_model, "features": X.columns.tolist(), "best_by": "f1"}, best_path, compress=3)
    best_meta = {
        "selected_model": best_name,
        "selected_by": "f1",
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "features": X.columns.tolist()
    }
    (Path(out_dir) / "best_model_metadata.json").write_text(json.dumps(best_meta, indent=2))

    print(f"\nSelected best model by CV mean: {best_name}")
    return results_df

if __name__ == "__main__":
    res= train_and_select("loan_data.csv", target_col="loan_status")
    